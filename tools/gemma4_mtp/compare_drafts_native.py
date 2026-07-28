#!/usr/bin/env python
"""
Official-contract HF comparison: same target, same inputs, official draft vs
your trained draft, using the NATIVE transformers Gemma4AssistantForCausalLM
forward contract (verified against transformers 5.9.0 source):

    draft.forward(inputs_embeds=cat[tok_embed*scale, target_hidden],
                  position_ids, shared_kv_states={type:(K,V)}) -> (logits, ...)

The draft ignores input_ids; it REQUIRES inputs_embeds + shared_kv_states.
Both drafts load as Gemma4AssistantForCausalLM (structure verified identical).
Only the weights differ -> isolates training vs engine.

Per position we compare each draft's greedy next-token vs the target's own
greedy next-token (argmax(target_lm_head(target_hidden))), scored over the
assistant-answer (supervised) region.

USAGE (server):
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/compare_drafts_native.py \
      --target   $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
      --official $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
      --trained  $AZURE_ML_INPUT_UKWDATA/maiprofile/mtp_26b/gemma4_mtp_exp1_hf/hf_iter_0002269 \
      --data     data/eval_layer1_delta_1000.jsonl --num-prompts 5 --max-len 2048
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

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Gemma4AssistantForCausalLM,
    )

    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)

    print("loading target ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16
    ).to(dev).eval()
    backbone = find_backbone(target)
    embed_w = backbone.embed_tokens.weight
    lm_head_w = target.get_output_embeddings().weight
    backbone_hidden = embed_w.shape[1]
    embed_scale = float(backbone_hidden ** 0.5)
    print(f"  backbone_hidden={backbone_hidden} embed_scale={embed_scale:.2f}")

    print("loading official + trained drafts (Gemma4AssistantForCausalLM) ...")
    d_off = Gemma4AssistantForCausalLM.from_pretrained(args.official, dtype=torch.bfloat16).to(dev).eval()
    d_trn = Gemma4AssistantForCausalLM.from_pretrained(args.trained, dtype=torch.bfloat16).to(dev).eval()

    lines = [l for l in open(args.data, encoding="utf-8") if l.strip()][: args.num_prompts]
    agg = {"official": [0, 0], "trained": [0, 0]}

    for i, line in enumerate(lines):
        rec = json.loads(line)
        convs = rec["conversations"]
        norm = [{"role": ("assistant" if m.get("role") in ("assistant", "model")
                          else m.get("role", "user")),
                 "content": m["content"]} for m in convs]

        # Full conversation tokens.
        full = tok.apply_chat_template(norm, tokenize=True, return_tensors="pt")
        if hasattr(full, "input_ids"):
            full = full.input_ids
        full = full[:, : args.max_len].to(dev)

        # Prompt-only tokens (everything up to the assistant turn + gen prompt).
        # The assistant answer is the region AFTER prompt_len -> that's where a
        # real speculative decoder actually drafts. Score ONLY there.
        prompt_msgs = [m for m in norm if m["role"] != "assistant"]
        pids = tok.apply_chat_template(prompt_msgs, tokenize=True,
                                       add_generation_prompt=True, return_tensors="pt")
        if hasattr(pids, "input_ids"):
            pids = pids.input_ids
        prompt_len = pids.shape[1]
        T = full.shape[1]
        if prompt_len >= T:
            print(f"  prompt {i}: SKIP (prompt_len {prompt_len} >= T {T})")
            continue
        ids = full
        pos = torch.arange(T, device=dev).unsqueeze(0)

        with torch.no_grad():
            tout = backbone(ids, use_cache=True, return_shared_kv_states=True)
            tgt_hidden = tout.last_hidden_state          # (1,T,2816) post-norm
            shared_kv = tout.shared_kv_states            # {type:(K,V)}

            tgt_pred = F.linear(tgt_hidden, lm_head_w).argmax(-1)  # (1,T)

            tok_emb = F.embedding(ids, embed_w) * embed_scale     # (1,T,2816)
            inputs_embeds = torch.cat([tok_emb, tgt_hidden], dim=-1)  # (1,T,5632)

            # Score only the assistant-answer region [prompt_len-1 : T-1]:
            # position p predicts token p+1; the first answer token is at
            # index prompt_len, predicted from position prompt_len-1.
            lo, hi = prompt_len - 1, T - 1
            for name, draft in (("official", d_off), ("trained", d_trn)):
                out = draft(inputs_embeds=inputs_embeds, position_ids=pos,
                            shared_kv_states=shared_kv)
                dpred = out.logits.argmax(-1)            # (1,T)
                dp = dpred[0, lo:hi]
                tp = tgt_pred[0, lo:hi]
                agg[name][0] += (dp == tp).sum().item()
                agg[name][1] += dp.numel()
        print(f"  prompt {i}: T={T} ans_tokens={hi-lo}  "
              f"official={agg['official'][0]}/{agg['official'][1]}  "
              f"trained={agg['trained'][0]}/{agg['trained'][1]}")

    print("=" * 60)
    print("Draft-vs-target greedy agreement (native HF contract, same target):")
    for name in ("official", "trained"):
        c, t = agg[name]
        print(f"  {name:9}: {c / max(t, 1):.4f}  ({c}/{t})")
    o = agg["official"][0] / max(agg["official"][1], 1)
    r = agg["trained"][0] / max(agg["trained"][1], 1)
    print("-" * 60)
    print(f"delta (official - trained) = {o - r:.4f}")
    print("trained << official => training moved the draft off distribution (HF).")
    print("trained ~= official => HF fine; gap is the vLLM engine path.")
    print("=" * 60)


if __name__ == "__main__":
    main()
