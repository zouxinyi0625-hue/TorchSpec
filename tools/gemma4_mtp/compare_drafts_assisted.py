#!/usr/bin/env python
"""
Official-style HF assisted-generation comparison: same target, same prompts,
official draft vs your trained draft. Uses transformers' native assisted
decoding (target.generate(assistant_model=draft)) -- the exact HF path, where
the draft is a standalone Gemma4AssistantForCausalLM (NOT the vLLM cross-model
KV-sharing path).

Measures per-draft acceptance via generation speed proxy: with assisted
decoding, more accepted draft tokens => fewer target forward calls => we count
matched tokens by comparing assisted output to the target's own greedy output
(they must be identical for greedy; the draft only affects speed, not result).
So we instead time / count target forward calls is not exposed; we use a direct
acceptance probe: for each position, does the draft's greedy next-token match
the target's greedy next-token, over a teacher-forced target-generated sequence.

USAGE (server):
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/compare_drafts_assisted.py \
      --target   $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
      --official $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
      --trained  $AZURE_ML_INPUT_UKWDATA/maiprofile/mtp_26b/gemma4_mtp_exp1_hf/hf_iter_0002269 \
      --data     data/eval_layer1_delta_1000.jsonl --num-prompts 5 --gen-len 128
"""
from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=1536)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.target)

    print("loading target ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype="auto", device_map="auto"
    ).eval()
    dev = target.device

    print("loading official + trained drafts ...")
    d_off = AutoModelForCausalLM.from_pretrained(
        args.official, dtype="auto", device_map="auto"
    ).eval()
    d_trn = AutoModelForCausalLM.from_pretrained(
        args.trained, dtype="auto", device_map="auto"
    ).eval()

    lines = [l for l in open(args.data, encoding="utf-8") if l.strip()][: args.num_prompts]

    # Acceptance proxy per draft: over a target-greedy-generated continuation,
    # count positions where draft-greedy(next) == target-greedy(next). Same
    # metric for both drafts -> directly comparable.
    agg = {"official": [0, 0], "trained": [0, 0]}

    for i, line in enumerate(lines):
        rec = json.loads(line)
        convs = rec["conversations"]
        msgs = [{"role": ("assistant" if m.get("role") in ("assistant", "model")
                          else m.get("role", "user")),
                 "content": m["content"]} for m in convs
                if m.get("role") in ("system", "user")]
        inputs = proc.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        ).to(dev)
        ids = inputs["input_ids"][:, : args.max_len]

        with torch.no_grad():
            # target greedy continuation
            gen = target.generate(input_ids=ids, max_new_tokens=args.gen_len,
                                  do_sample=False)
            seq = gen  # (1, L)
            L = seq.shape[1]

            # target's own next-token greedy over the full seq
            tout = target(seq)
            tgt_pred = tout.logits.argmax(-1)  # (1, L)

            for name, draft in (("official", d_off), ("trained", d_trn)):
                dout = draft(seq)
                dpred = dout.logits.argmax(-1)  # (1, L)
                # score only the generated continuation region
                start = ids.shape[1]
                dp = dpred[0, start - 1 : L - 1]
                tp = tgt_pred[0, start - 1 : L - 1]
                agg[name][0] += (dp == tp).sum().item()
                agg[name][1] += dp.numel()
        print(f"  prompt {i}: L={L}  official={agg['official'][0]}/{agg['official'][1]}  "
              f"trained={agg['trained'][0]}/{agg['trained'][1]}")

    print("=" * 60)
    print("Draft-vs-target greedy agreement (same target, HF, generated region):")
    for name in ("official", "trained"):
        c, t = agg[name]
        print(f"  {name:9}: {c / max(t, 1):.4f}  ({c}/{t})")
    o = agg["official"][0] / max(agg["official"][1], 1)
    r = agg["trained"][0] / max(agg["trained"][1], 1)
    print("-" * 60)
    print(f"delta (official - trained) = {o - r:.4f}")
    print("trained << official => training moved the draft off distribution (HF).")
    print("trained ~= official => HF is fine; the gap is the vLLM engine path.")
    print("=" * 60)


if __name__ == "__main__":
    main()
