#!/usr/bin/env python
"""
PARITY PROBE: is the trained draft self-consistent on the TRAINING path?

The deployment gap (vLLM accept ~0.43 vs training eval ~0.91, uniform halving
incl pos0) is NOT explained by fp8/loading/embed_scale/norm (all ruled out).
This probe answers the decisive question: fed the SAME real
(input_ids, target_last_hidden, shared_kv) that TRAINING used, does the draft's
step-0 prediction agree with target-greedy at ~0.9?

  - If YES (pos0 top-1 agreement ~0.9): the TRAINING path is self-consistent.
    The gap is entirely in what vLLM FEEDS the draft at deploy (different
    target_hidden / shared_kv than training). Next: dump vLLM's actual draft
    input and diff.
  - If NO (pos0 ~0.4): the draft is NOT actually good on a real single-step
    rollout; the "training eval 0.91" is a different-harness artifact (e.g.
    teacher-force + cached hidden hides a real weakness). Next: audit the
    training eval metric.

It reproduces the training forward EXACTLY (torchspec/models/gemma4_mtp.py:
139-160): prev_hidden = target_last_hidden, cur_token = input_ids[t],
embed_scale = sqrt(backbone_hidden=2816), inputs_embeds = concat[
target_embed(cur_token)*scale, prev_hidden], then draft.forward(...).

RUN ON THE SERVER (1 GPU), from the TorchSpec root:
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/probe_draft_parity.py \
      --target    /tmp/models/gemma4/text_only \
      --assistant $AZURE_ML_INPUT_UKWDATA/maiprofile/mtp_26b/gemma4_mtp_exp1_hf/hf_iter_0002269 \
      --data      data/eval_layer1_delta_1000.jsonl \
      --max-len 2048
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F


def find_backbone_and_norm(m):
    import torch.nn as nn
    cands, seen = [m], set()
    while cands:
        mod = cands.pop(0)
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        if (hasattr(mod, "norm") and isinstance(getattr(mod, "norm"), nn.Module)
                and hasattr(mod, "layers")):
            return mod, mod.norm
        for _n, c in mod.named_children():
            cands.append(c)
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--assistant", required=True, help="Trained draft (HF assistant) dir")
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Gemma4AssistantForCausalLM,
    )

    dev = args.device
    print(f"Loading target {args.target} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(dev).eval()
    backbone, _ = find_backbone_and_norm(target)

    print(f"Loading trained draft {args.assistant} ...", flush=True)
    draft = Gemma4AssistantForCausalLM.from_pretrained(
        args.assistant, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(dev).eval()

    # TARGET embedding table (scaled convention) + lm_head (tied for Gemma4).
    target_embed = backbone.embed_tokens.weight            # (V, 2816)
    backbone_hidden = target_embed.shape[1]
    embed_scale = torch.tensor(backbone_hidden ** 0.5, dtype=target_embed.dtype, device=dev)

    # Build one real prompt.
    with open(args.data, encoding="utf-8") as f:
        rec = json.loads(f.readline())
    convs = rec["conversations"]
    sys_c = next((t["content"] for t in convs if t["role"] == "system"), "")
    usr_c = next((t["content"] for t in convs if t["role"] == "user"), "")
    text = (sys_c + "\n\n" + usr_c) if sys_c else usr_c
    ids = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len).input_ids.to(dev)
    B, T = ids.shape
    print(f"prompt tokens: {T}", flush=True)

    # Target forward -> last_hidden (what training fed as prev_hidden) + shared_kv + greedy.
    with torch.no_grad():
        bb_out = backbone(ids, use_cache=True, return_shared_kv_states=True,
                          output_hidden_states=False)
        target_last_hidden = bb_out.last_hidden_state             # (B, T, 2816)
        shared_kv = bb_out.shared_kv_states
        tgt_logits = target(ids, use_cache=False).logits          # (B, T, V)
        target_greedy = tgt_logits.argmax(-1)                     # (B, T)

    if shared_kv is None:
        raise SystemExit("target did not return shared_kv_states; check return_shared_kv_states support")

    # Reproduce the training step-0 draft input EXACTLY (gemma4_mtp.py:139-162).
    # prev_hidden = target_last_hidden ; cur_token = input_ids[t].
    cur_token = ids.clamp(0, target_embed.shape[0] - 1)
    tok_embed = F.embedding(cur_token, target_embed) * embed_scale     # (B, T, 2816), scaled
    inputs_embeds = torch.cat([tok_embed, target_last_hidden], dim=-1)  # (B, T, 5632)
    position_ids = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)

    print("norms: tok_embed(scaled)={:.2f}  prev_hidden={:.2f}  concat={:.2f}".format(
        tok_embed[0].float().norm(dim=-1).mean().item(),
        target_last_hidden[0].float().norm(dim=-1).mean().item(),
        inputs_embeds[0].float().norm(dim=-1).mean().item(),
    ), flush=True)

    with torch.no_grad():
        out = draft(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            shared_kv_states=shared_kv,
        )
    draft_logits = out.logits                                    # (B, T, V)
    draft_greedy = draft_logits.argmax(-1)                       # (B, T)

    # Step-0 label alignment (gemma4_mtp.py: step k supervises target_greedy[t+k+1]).
    # For pos0 (k=0): draft prediction at t should match target_greedy at t+1.
    dr = draft_greedy[:, :-1]          # prediction made at position t
    tg = target_greedy[:, 1:]          # target's greedy next token
    # Only score supervised (non-pad) positions; here all real tokens.
    agree = (dr == tg).float().mean().item()

    print("\n================= DRAFT PARITY PROBE (step-0) =================")
    print(f"pos0 top-1 agreement (draft argmax == target greedy, shift+1): {agree:.4f}")
    print("-------------------------------------------------------------")
    if agree > 0.7:
        print("VERDICT: draft is SELF-CONSISTENT on the training path (~training eval).")
        print("  => The deployment gap is in WHAT vLLM FEEDS the draft, not the draft")
        print("     or its forward. Next: dump vLLM's real target_hidden/shared_kv for")
        print("     the SAME prompt and diff against these (norms + argmax).")
    elif agree < 0.55:
        print("VERDICT: draft is NOT self-consistent even on the training path.")
        print("  => 'training eval 0.91' is a different-harness artifact (teacher-force +")
        print("     cached hidden). Audit the training eval metric / label alignment.")
    else:
        print("VERDICT: intermediate — re-run on more prompts; check label shift (k vs k+1).")
    print("=============================================================")


if __name__ == "__main__":
    main()
