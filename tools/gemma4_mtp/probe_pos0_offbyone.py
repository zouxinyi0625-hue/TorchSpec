#!/usr/bin/env python
"""
Prove/disprove the pos0 off-by-one: at position t, does the draft consume
token_t (training's cur_token=input_ids) or token_{t+1} (vLLM's
input_ids[:n-1]=target_token_ids[1:])?

Uses the OFFICIAL draft (known ~84% pos0 in vLLM) as ground truth. Pure HF,
native Gemma4AssistantForCausalLM contract (inputs_embeds + shared_kv). For each
supervised answer position we form the draft input two ways and compare its
greedy argmax against the target's own greedy next-token (the accept proxy):

  A (training):  inputs_embeds = cat[ embed(token_t)   * scale, hidden_t ]
  B (vLLM):      inputs_embeds = cat[ embed(token_{t+1})* scale, hidden_t ]

Whichever matches the target-greedy far better is what vLLM actually feeds. If B
>> A, the off-by-one is real and training feeds the wrong token at pos0.

USAGE:
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/probe_pos0_offbyone.py \
      --target   $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
      --official $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
      --data data/eval_layer1_delta_1000.jsonl --num-prompts 20 --max-len 2048
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
    ap.add_argument("--data", required=True)
    ap.add_argument("--num-prompts", type=int, default=20)
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
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, trust_remote_code=True
    ).to(dev).eval()
    backbone = find_backbone(target)
    embed_w = backbone.embed_tokens.weight
    lm_head_w = target.get_output_embeddings().weight
    bh = embed_w.shape[1]
    scale = float(bh ** 0.5)

    draft = Gemma4AssistantForCausalLM.from_pretrained(
        args.official, dtype=torch.bfloat16).to(dev).eval()

    lines = [l for l in open(args.data, encoding="utf-8") if l.strip()][: args.num_prompts]
    agg = {"A_token_t": [0, 0], "B_token_t+1": [0, 0]}

    for i, line in enumerate(lines):
        rec = json.loads(line)
        norm = [{"role": ("system" if m.get("role") == "system" else
                          ("assistant" if m.get("role") in ("assistant", "model") else "user")),
                 "content": m["content"]} for m in rec["conversations"]]
        full = tok.apply_chat_template(norm, tokenize=True, return_tensors="pt")
        if hasattr(full, "input_ids"):
            full = full.input_ids
        full = full[:, : args.max_len].to(dev)
        prompt_msgs = [m for m in norm if m["role"] != "assistant"]
        pids = tok.apply_chat_template(prompt_msgs, tokenize=True,
                                       add_generation_prompt=True, return_tensors="pt")
        if hasattr(pids, "input_ids"):
            pids = pids.input_ids
        prompt_len = pids.shape[1]
        T = full.shape[1]
        if prompt_len >= T - 2:
            continue
        ids = full
        pos = torch.arange(T, device=dev).unsqueeze(0)

        with torch.no_grad():
            tout = backbone(ids, use_cache=True, return_shared_kv_states=True)
            hidden = tout.last_hidden_state              # (1,T,2816) hidden_t
            shared_kv = tout.shared_kv_states
            tgt_pred = F.linear(hidden, lm_head_w).argmax(-1)  # (1,T) target-greedy next

            emb = F.embedding(ids, embed_w) * scale      # (1,T,2816) embed(token_t)
            # A: token_t at position t (training's cur_token=input_ids)
            emb_A = emb
            # B: token_{t+1} at position t (vLLM input_ids[:n-1]=target_token_ids[1:])
            emb_B = torch.zeros_like(emb)
            emb_B[:, :-1, :] = emb[:, 1:, :]             # shift: pos t gets embed(token_{t+1})

            for name, tok_half in (("A_token_t", emb_A), ("B_token_t+1", emb_B)):
                inp = torch.cat([tok_half, hidden], dim=-1)   # (1,T,5632)
                out = draft(inputs_embeds=inp, position_ids=pos,
                            shared_kv_states=shared_kv)
                dpred = out.logits.argmax(-1)            # (1,T)
                lo, hi = prompt_len - 1, T - 2
                dp = dpred[0, lo:hi]
                if name == "B_token_t+1":
                    # draft at pos t was fed token_{t+1} and hidden_t, so it
                    # predicts token_{t+2}; compare against target-greedy at t+1
                    # (= token_{t+2}), NOT at t.
                    tp = tgt_pred[0, lo + 1:hi + 1]
                else:
                    # A fed token_t -> predicts token_{t+1} = target-greedy at t
                    tp = tgt_pred[0, lo:hi]
                agg[name][0] += (dp == tp).sum().item()
                agg[name][1] += dp.numel()
        print(f"  prompt {i}: T={T}  A={agg['A_token_t'][0]}/{agg['A_token_t'][1]}  "
              f"B={agg['B_token_t+1'][0]}/{agg['B_token_t+1'][1]}")

    print("=" * 60)
    print("Official draft pos0 agreement vs target-greedy (which token vLLM feeds):")
    for name in ("A_token_t", "B_token_t+1"):
        c, t = agg[name]
        print(f"  {name:12}: {c / max(t, 1):.4f}  ({c}/{t})")
    a = agg["A_token_t"][0] / max(agg["A_token_t"][1], 1)
    b = agg["B_token_t+1"][0] / max(agg["B_token_t+1"][1], 1)
    print("-" * 60)
    print(f"A (training, token_t) = {a:.4f}   B (vLLM, token_t+1) = {b:.4f}")
    print("vLLM official pos0 ~= 0.84. Whichever is closer is what vLLM feeds.")
    if b > a + 0.1:
        print("=> B wins: OFF-BY-ONE CONFIRMED. Training feeds token_t but vLLM")
        print("   feeds token_{t+1}. Fix training's cur_token to shift +1.")
    elif a > b + 0.1:
        print("=> A wins: training's token is correct; off-by-one is NOT the cause.")
    else:
        print("=> inconclusive; neither near 0.84 -> probe/contract issue, inspect.")
    print("=" * 60)


if __name__ == "__main__":
    main()
