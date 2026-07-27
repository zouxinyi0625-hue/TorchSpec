#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
"""
Convert a MaiProfile layer eval jsonl (schema: user_id, prompt_hash,
conversations=[{role,content}...]) into the flat {"prompt": ...} jsonl that
vllm-msn/benchmarks/gemma4_12b_fp8/bench_offline_align.py expects.

The benchmark's render_chat() wraps each prompt as a single user turn:
    apply_chat_template([{"role":"user","content":<prompt>}], add_generation_prompt=True)
So we fold the ORIGINAL system + user turns of each conversation into one
prompt string (system instruction first, then the user signal block). The
assistant turn (ground-truth answer) is dropped — offline accept only needs
the prompt; the draft generates and the target verifies live.

USAGE:
    python tools/gemma4_mtp/conversations_to_prompts.py \
        --in  data/eval_layer1_delta_1000.jsonl \
        --out data/eval_layer1_delta_prompts.jsonl

Then benchmark with --dataset-path data/eval_layer1_delta_prompts.jsonl
"""
import argparse
import json


def build_prompt(conversations) -> str | None:
    """Fold system+user turns into one prompt string; drop assistant.

    Output format matches the reference sc1_delta_v2.jsonl exactly (verified by
    reconstructing a real record byte-for-byte):
        "[SYSTEM]\n" + <system> + "\n\n" + <user>
    No [USER] marker, no trailing generation prompt. The benchmark's
    render_chat() then wraps this whole string as a single user turn.
    """
    system_parts = []
    user_parts = []
    for turn in conversations:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        # assistant / other roles: ignored (draft generates these)
    if not user_parts:
        return None  # no user turn -> nothing to prompt with
    system_block = "\n\n".join(system_parts)
    user_block = "\n\n".join(user_parts)
    if system_block:
        return f"[SYSTEM]\n{system_block}\n\n{user_block}"
    return user_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap #records (0 = all)")
    args = ap.parse_args()

    total = written = skipped = 0
    with open(args.inp, encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                skipped += 1
                continue
            convs = rec.get("conversations")
            if not convs:
                skipped += 1
                continue
            prompt = build_prompt(convs)
            if not prompt:
                skipped += 1
                continue
            fout.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break

    print(f"scanned {total}, wrote {written}, skipped {skipped} -> {args.out}")


if __name__ == "__main__":
    main()
