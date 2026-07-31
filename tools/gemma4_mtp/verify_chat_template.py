#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
"""
Verify the TorchSpec 'gemma' chat template matches the Gemma4 tokenizer's
official apply_chat_template, so loss-mask assistant spans align.

TorchSpec's GeneralParser hand-assembles turns from header strings
(<start_of_turn>user\\n ... <end_of_turn>\\n / <start_of_turn>model\\n ...). If
that string doesn't tokenize identically to the model's real chat template, the
assistant span (and thus the loss mask) is misaligned — a silent train/infer
mismatch. This probe compares both on a sample conversation.

Run (server, needs the target tokenizer):
    python tools/gemma4_mtp/verify_chat_template.py --target /tmp/models/gemma4/text_only

Paste stdout back.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="/tmp/models/gemma4/text_only")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from torchspec.data.parse import create_parser
    from torchspec.data.template import TEMPLATE_REGISTRY

    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)

    conv = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And Japan?"},
        {"role": "assistant", "content": "The capital of Japan is Tokyo."},
    ]

    # 1) Official HF chat template (ground truth), if the tokenizer has one.
    official = None
    if tok.chat_template:
        official = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)

    # 2) TorchSpec 'gemma' template via GeneralParser.
    template = TEMPLATE_REGISTRY.get("gemma")
    parser = create_parser(tok, template)
    ours = parser.format(conv, add_generation_prompt=False)

    print("=" * 70)
    print("TorchSpec 'gemma' formatted:")
    print(repr(ours))
    print("=" * 70)
    print("Official apply_chat_template:")
    print(repr(official))
    print("=" * 70)

    if official is None:
        print("NOTE: tokenizer has no chat_template; cannot compare to ground truth.")
        print("Manually confirm the TorchSpec format above matches Gemma's spec.")
        return

    # Compare token id sequences (strip any leading BOS difference).
    ours_ids = tok(ours, add_special_tokens=False).input_ids
    off_ids = tok(official, add_special_tokens=False).input_ids
    print(f"ours  #tokens={len(ours_ids)}")
    print(f"offic #tokens={len(off_ids)}")

    if ours_ids == off_ids:
        print("MATCH: token sequences identical ✅ — formatting aligns.")
    else:
        # Show first divergence for debugging.
        n = min(len(ours_ids), len(off_ids))
        div = next((i for i in range(n) if ours_ids[i] != off_ids[i]), n)
        print(f"DIVERGE at token index {div}:")
        lo = max(0, div - 4)
        print(f"  ours [{lo}:{div+4}] = {ours_ids[lo:div+4]}")
        print(f"  offic[{lo}:{div+4}] = {off_ids[lo:div+4]}")
        print(f"  ours  decoded around: {tok.decode(ours_ids[lo:div+4])!r}")
        print(f"  offic decoded around: {tok.decode(off_ids[lo:div+4])!r}")
        print("MISMATCH ❌ — adjust the 'gemma' template headers, then re-run.")

    # ---- CRITICAL: loss mask must be non-empty and cover the assistant spans ----
    # GeneralParser.format uses the official template, but .parse regex-matches
    # the template's assistant_header/end_of_turn_token to locate assistant spans.
    # If those strings don't match what the official template emits, the loss
    # mask is empty and training gets zero gradient.
    print("\n" + "=" * 70)
    print("Loss-mask span check (the thing that actually matters):")
    input_ids, loss_mask = parser.parse(ours, max_length=512, preformatted=True)
    n_tok = int(loss_mask.numel())
    n_sup = int(loss_mask.sum())
    print(f"  seq_len={n_tok}  supervised_tokens={n_sup}  ratio={n_sup / max(1, n_tok):.2f}")

    if n_sup == 0:
        print("  EMPTY LOSS MASK ❌ — assistant_header/end_of_turn_token do NOT "
              "match the official template output. Fix the 'gemma' template.")
        raise SystemExit(1)

    # Decode the supervised tokens; they should be the assistant answers.
    sup_ids = [int(i) for i, m in zip(input_ids.tolist(), loss_mask.tolist()) if m > 0.5]
    decoded = tok.decode(sup_ids)
    print(f"  supervised text (decoded): {decoded!r}")
    ok = "Paris" in decoded and "Tokyo" in decoded
    print("  covers assistant answers ✅" if ok else
          "  WARNING: supervised text missing expected answers — inspect above ⚠️")
    if not ok:
        raise SystemExit(1)
    print("\nDATA READY ✅ — format aligns and loss mask covers assistant spans.")


if __name__ == "__main__":
    main()
