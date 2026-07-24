#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
"""
Dump the EXACT MTP drafting loop Google uses to drive the assistant, to a file.

The grep already found the key class:
    generation/candidate_generator.py :: SinglePositionMultiTokenCandidateGenerator
    "predicting multiple draft tokens from a single token position using MTP"

This is the ground-truth training contract. We dump its FULL source (plus the
assistant forward and the MTP output dataclass) to a file so nothing truncates.

Run:
    python tools/gemma4_mtp/dump_mtp_loop.py            # writes /tmp/gemma4_mtp_loop.txt
    python tools/gemma4_mtp/dump_mtp_loop.py --out /some/path.txt

Then send back the file (it's small, a few hundred lines).
"""
import argparse
import inspect
import io
import os


def sec(w, title):
    w.write("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78 + "\n")


def dump_obj(w, obj, title):
    sec(w, title)
    try:
        w.write(inspect.getsource(obj))
    except (OSError, TypeError) as e:
        w.write(f"<source unavailable: {e}>\n")


def dump_file_slice(w, path, start, end, title):
    """Dump raw lines [start, end] (1-indexed) from a file."""
    sec(w, f"{title}  ({os.path.basename(path)} lines {start}-{end})")
    if not os.path.exists(path):
        w.write(f"<file not found: {path}>\n")
        return
    with open(path) as f:
        lines = f.readlines()
    for i in range(start - 1, min(end, len(lines))):
        w.write(f"{i + 1:5d}| {lines[i]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/gemma4_mtp_loop.txt")
    ap.add_argument("--assistant", default="/tmp/models/gemma4/assistant")
    args = ap.parse_args()

    import transformers
    from transformers.generation import candidate_generator as cg

    w = io.StringIO()
    w.write("Gemma4 MTP drafting loop — ground truth for TorchSpec training\n")
    w.write(f"transformers @ {transformers.__version__}\n")

    # 1. The MTP candidate generator — the whole class, all methods.
    cls = getattr(cg, "SinglePositionMultiTokenCandidateGenerator", None)
    if cls is not None:
        dump_obj(w, cls, "SinglePositionMultiTokenCandidateGenerator (FULL CLASS)")
    else:
        # Fallback: raw line slice around the grep hit.
        dump_file_slice(w, cg.__file__, 1230, 1460,
                        "SinglePositionMultiTokenCandidateGenerator (raw slice)")

    # 2. The module-level helper that builds shared_kv_states overrides.
    dump_file_slice(w, cg.__file__, 1265, 1300,
                    "shared_kv_states model_kwargs_overrides + init")

    # 3. Assistant forward + MTP output dataclass, from live import.
    try:
        from transformers import Gemma4AssistantForCausalLM as A
        dump_obj(w, A.forward, "Gemma4AssistantForCausalLM.forward")
        dump_obj(w, A.create_attention_masks,
                 "Gemma4AssistantForCausalLM.create_attention_masks")
        if getattr(A, "masked_embedding", None) is not None:
            pass
    except Exception as e:
        w.write(f"\n<assistant import failed: {e}>\n")

    # 4. The assistant modeling file's create_attention_masks region (raw), in
    #    case it lives on the inner model.
    try:
        import transformers.models.gemma4_assistant.modeling_gemma4_assistant as m
        dump_file_slice(w, m.__file__, 195, 260,
                        "modeling_gemma4_assistant create_attention_masks (raw)")
    except Exception as e:
        w.write(f"\n<assistant modeling slice failed: {e}>\n")

    text = w.getvalue()
    with open(args.out, "w") as f:
        f.write(text)
    print(f"Wrote {len(text.splitlines())} lines to {args.out}")
    print("Send back that file.")


if __name__ == "__main__":
    main()
