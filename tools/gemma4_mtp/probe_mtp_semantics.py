#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
"""
Reverse-engineer how Gemma4 drives its assistant for multi-step MTP, straight
from the HF source — so training replicates Google's ACTUAL forward contract,
not a guess.

We need to answer, from the real code:
  1. Does one assistant.forward predict ONE future token (then recurse by
     feeding post_projection's last_hidden_state back as prev_hidden), or does
     it emit N future tokens in a single pass?
  2. What exactly is fed as `inputs_embeds` on step t>0 — is prev_hidden the
     target's hidden, or the assistant's own post_projection output?
  3. How are shared_kv_states reused/extended across the drafted block?
  4. Position ids / attention mask handling across the drafted tokens.

Strategy: locate the modeling_gemma4*.py file(s), print the assistant class,
the CausalLM wrapper, and — crucially — Gemma4's generation/assistant-drafting
glue (the candidate generator or assisted-decoding hook). No weights loaded.

Run:
    python tools/gemma4_mtp/probe_mtp_semantics.py --assistant /tmp/models/gemma4/assistant

Paste stdout back.
"""
import argparse
import importlib
import inspect
import os
import re


def dump_source(obj, tag, max_lines=400):
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError) as e:
        print(f"\n[{tag}] source unavailable: {e}")
        return
    lines = src.splitlines()
    print(f"\n{'=' * 70}\n[{tag}]  ({len(lines)} lines"
          f"{', truncated' if len(lines) > max_lines else ''})\n{'=' * 70}")
    print("\n".join(lines[:max_lines]))


def grep_file(path, patterns, ctx=3, tag=""):
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = f.readlines()
    print(f"\n{'#' * 70}\n# GREP {tag or path}\n{'#' * 70}")
    for pat in patterns:
        rx = re.compile(pat)
        hits = [i for i, ln in enumerate(lines) if rx.search(ln)]
        if hits:
            print(f"\n--- /{pat}/ ({len(hits)} hits) ---")
        for i in hits[:8]:
            lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
            for j in range(lo, hi):
                mark = ">>" if j == i else "  "
                print(f"{mark}{j + 1:5d}| {lines[j].rstrip()}")
            print("    ...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assistant", default="/tmp/models/gemma4/assistant")
    args = ap.parse_args()

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.assistant, trust_remote_code=True)
    print(f"assistant model_type={cfg.model_type} "
          f"arch={cfg.architectures} use_ordered_embeddings="
          f"{getattr(cfg, 'use_ordered_embeddings', '?')} "
          f"num_centroids={getattr(cfg, 'num_centroids', '?')} "
          f"centroid_top_k={getattr(cfg, 'centroid_intermediate_top_k', '?')}")

    # --- find the modeling module(s) for gemma4 ---
    import transformers
    tr_dir = os.path.dirname(transformers.__file__)
    candidates = []
    for root, _, files in os.walk(os.path.join(tr_dir, "models")):
        for fn in files:
            if re.search(r"(modeling|generation).*gemma4", fn) or "gemma4" in root and fn.startswith("modeling"):
                candidates.append(os.path.join(root, fn))
    print(f"\n[FILES] gemma4 modeling/generation files:")
    for c in candidates:
        print(f"  {c}")

    # --- classes we care about, dumped from live imports ---
    for cls_name in [
        "Gemma4AssistantForCausalLM",
        "Gemma4AssistantModel",
        "Gemma4ForCausalLM",
    ]:
        try:
            cls = getattr(importlib.import_module("transformers"), cls_name, None)
            if cls is None:
                print(f"\n[{cls_name}] not exported at top level")
                continue
            # forward + any masked_embedding / centroid helpers
            if hasattr(cls, "forward"):
                dump_source(cls.forward, f"{cls_name}.forward", max_lines=120)
            for helper in ["masked_embedding", "create_attention_masks",
                           "_assisted_ids", "prepare_inputs_for_generation"]:
                if hasattr(cls, helper):
                    dump_source(getattr(cls, helper), f"{cls_name}.{helper}", max_lines=80)
        except Exception as e:
            print(f"\n[{cls_name}] error: {e}")

    # --- grep the modeling + assisted-generation glue for the drafting loop ---
    for c in candidates:
        grep_file(
            c,
            patterns=[
                r"assistant", r"shared_kv_states", r"post_projection",
                r"pre_projection", r"last_hidden_state", r"prev_hidden",
                r"num_.*(nextn|mtp|future|assistant|draft)", r"for .* in range",
                r"centroid", r"masked_embedding", r"ordered_embed",
            ],
            tag=os.path.basename(c),
        )

    # --- also grep the candidate generator (assisted decoding) ---
    cand_gen = os.path.join(tr_dir, "generation", "candidate_generator.py")
    grep_file(
        cand_gen,
        patterns=[
            r"class .*CandidateGenerator", r"assistant_model",
            r"shared_kv_states", r"get_candidates", r"num_assistant_tokens",
            r"inputs_embeds", r"last_hidden_state", r"post_projection",
        ],
        tag="generation/candidate_generator.py",
    )

    print("\n# DONE. Paste everything above back.")


if __name__ == "__main__":
    main()
