#!/usr/bin/env python
"""
vLLM OFFLINE per-position acceptance comparison: same target, same prompts,
official draft vs your trained draft. This is the deploy-truth engine (you only
care about vLLM). Uses vllm.LLM(...).get_metrics() to read:
  vllm:spec_decode_num_drafts
  vllm:spec_decode_num_draft_tokens
  vllm:spec_decode_num_accepted_tokens
  vllm:spec_decode_num_accepted_tokens_per_pos   (per-position accept)

Runs one draft per process invocation (vLLM builds one engine per LLM). Pass
--assistant to pick which draft; run twice (official, trained) and compare, or
use --assistant twice via the wrapper at the bottom.

USAGE (server, single invocation per draft):
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/vllm_offline_accept.py \
      --target    $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
      --assistant $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
      --data data/eval_layer1_delta_1000.jsonl --num-prompts 20 \
      --num-spec-tokens 5 --max-tokens 256 --tp 2 --tag official

Then again with --assistant <your hf_iter> --tag trained, and diff the two.
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--assistant", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--num-spec-tokens", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tag", default="draft")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)

    # Build prompts: system+user only, model generates the answer (real drafting).
    # Skip prompts that don't leave room for output within max_model_len.
    budget = args.max_model_len - args.max_tokens - 8
    prompts, skipped = [], 0
    for line in open(args.data, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        msgs = [{"role": ("system" if m.get("role") == "system" else "user"),
                 "content": m["content"]}
                for m in rec["conversations"] if m.get("role") in ("system", "user")]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        n_tok = len(tok(text).input_ids)
        if n_tok > budget:
            skipped += 1
            continue
        prompts.append(text)
        if len(prompts) >= args.num_prompts:
            break
    print(f"prompts kept={len(prompts)} skipped_too_long={skipped} (budget={budget} tok)")

    llm = LLM(
        model=args.target,
        speculative_config={"model": args.assistant,
                            "num_speculative_tokens": args.num_spec_tokens},
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens,
                        top_p=(0.95 if args.temperature > 0 else 1.0))
    llm.generate(prompts, sp)

    # Read spec-decode metrics.
    drafts = draft_toks = accepted = 0
    per_pos = {}
    for m in llm.get_metrics():
        n = m.name
        v = getattr(m, "value", None)
        if n == "vllm:spec_decode_num_drafts":
            drafts = v
        elif n == "vllm:spec_decode_num_draft_tokens":
            draft_toks = v
        elif n == "vllm:spec_decode_num_accepted_tokens":
            accepted = v
        elif n == "vllm:spec_decode_num_accepted_tokens_per_pos":
            # value may be a dict/list keyed by position
            per_pos = getattr(m, "values", None) or v

    print("=" * 60)
    print(f"[{args.tag}] vLLM offline spec-decode acceptance")
    print(f"  assistant       = {args.assistant}")
    print(f"  num_drafts      = {drafts}")
    print(f"  num_draft_tokens= {draft_toks}")
    print(f"  num_accepted    = {accepted}")
    if draft_toks:
        print(f"  accept_rate     = {accepted / draft_toks:.4f}")
    if drafts:
        print(f"  accept_len      = {accepted / drafts + 1:.2f}")
    print(f"  per_pos_accepted= {per_pos}")
    print("=" * 60)


if __name__ == "__main__":
    main()
