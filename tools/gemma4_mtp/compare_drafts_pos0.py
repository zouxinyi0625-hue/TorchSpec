#!/usr/bin/env python
"""
Head-to-head pos0 comparison, pure HF, SAME target + SAME inputs:
  official draft ($.../models/assistant)  vs  your trained draft (hf_iter_...)

Both drafts use IDENTICAL forward math (loaded as HF gemma4_assistant, which
implements pre_projection -> Q-only layers reading target shared_kv ->
post_projection -> lm_head). The ONLY variable is the draft weights. This
isolates: did training move the draft off the deploy-time behavior?

Per position we compare each draft's step-0 argmax against the target's own
next-token argmax (argmax(lm_head(target_last_hidden[t]))) -- the same acceptance
proxy training uses. Scored on supervised (assistant-answer) tokens only.

USAGE (server, 1 GPU enough for the drafts; target is the big one):
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/compare_drafts_pos0.py \
      --target    $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
      --official  $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
      --trained   $AZURE_ML_INPUT_UKWDATA/maiprofile/mtp_26b/gemma4_mtp_exp1_hf/hf_iter_0002269 \
      --data      data/eval_layer1_delta_1000.jsonl --num-prompts 5 --max-len 2048
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F


def find_backbone(m):
    import torch.nn as nn
    q, seen = [m], set()
    while q:
        mod = q.pop(0)
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        if (hasattr(mod, "norm") and isinstance(getattr(mod, "norm"), nn.Module)
                and hasattr(mod, "layers")):
            return mod
        for _n, c in mod.named_children():
            q.append(c)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)

    print("loading target ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(dev).eval()
    backbone = find_backbone(target)
    embed_w = backbone.embed_tokens.weight            # (V, 2816)
    lm_head_w = target.get_output_embeddings().weight  # tied -> (V, 2816)
    backbone_hidden = embed_w.shape[1]
    embed_scale = float(backbone_hidden ** 0.5)
    print(f"  backbone_hidden={backbone_hidden} embed_scale={embed_scale:.2f}")

    def load_draft(path):
        return AutoModel.from_pretrained(
            path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(dev).eval()

    print("loading official draft ...")
    d_off = load_draft(args.official)
    print("loading trained draft ...")
    d_trn = load_draft(args.trained)

    lines = [l for l in open(args.data, encoding="utf-8") if l.strip()][: args.num_prompts]

    def run_draft(draft, inputs_embeds, position_ids, shared_kv):
        out = draft(inputs_embeds=inputs_embeds, position_ids=position_ids,
                    shared_kv_states=shared_kv, use_cache=False)
        return out.logits  # (B, T, V)

    agg = {"official": [0, 0], "trained": [0, 0]}  # [correct, total]

    for i, line in enumerate(lines):
        rec = json.loads(line)
        convs = rec["conversations"]
        msgs = [{"role": ("assistant" if m.get("role") in ("assistant", "model")
                          else m.get("role", "user")),
                 "content": m["content"]} for m in convs]
        ids = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt")
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        ids = ids[:, : args.max_len].to(dev)
        T = ids.shape[1]
        pos = torch.arange(T, device=dev).unsqueeze(0)

        with torch.no_grad():
            tout = backbone(ids, use_cache=True, return_shared_kv_states=True,
                            output_hidden_states=False)
            tgt_hidden = tout.last_hidden_state            # (1, T, 2816) post-norm
            shared_kv = tout.shared_kv_states              # {type: (K,V)}

            # target's own next-token argmax (acceptance proxy)
            tgt_logits = F.linear(tgt_hidden, lm_head_w)   # (1, T, V)
            tgt_pred = tgt_logits.argmax(-1)               # (1, T)

            # draft step-0 inputs: concat[ tok_embed(ids)*scale , tgt_hidden ]
            tok_emb = F.embedding(ids, embed_w) * embed_scale  # (1, T, 2816)
            inputs_embeds = torch.cat([tok_emb, tgt_hidden], dim=-1)  # (1, T, 5632)

            for name, draft in (("official", d_off), ("trained", d_trn)):
                dl = run_draft(draft, inputs_embeds, pos, shared_kv)  # (1,T,V)
                dpred = dl.argmax(-1)                        # (1, T)
                # score all positions (both drafts scored identically)
                correct = (dpred == tgt_pred).sum().item()
                agg[name][0] += correct
                agg[name][1] += dpred.numel()
        print(f"  prompt {i}: T={T}  official={agg['official'][0]}/{agg['official'][1]}  "
              f"trained={agg['trained'][0]}/{agg['trained'][1]}")

    print("=" * 60)
    print("POS0 agreement vs target-greedy (same target, same inputs, HF):")
    for name in ("official", "trained"):
        c, t = agg[name]
        print(f"  {name:9}: {c/max(t,1):.4f}  ({c}/{t})")
    o = agg["official"][0] / max(agg["official"][1], 1)
    r = agg["trained"][0] / max(agg["trained"][1], 1)
    print("-" * 60)
    print(f"delta (official - trained) = {o - r:.4f}")
    print("If trained << official HERE (pure HF, same target): training moved the")
    print("draft off the target distribution -> it's a TRAINING problem, retrain.")
    print("If trained ~= official here but vLLM deploy differs: it's an ENGINE issue.")
    print("=" * 60)


if __name__ == "__main__":
    main()
