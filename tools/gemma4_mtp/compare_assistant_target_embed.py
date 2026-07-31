#!/usr/bin/env python
"""
Compare the OFFICIAL Gemma4 assistant's embedding / lm_head against the TARGET
model's, to decide whether training should FREEZE those layers.

Rationale: we warm-start training from the official assistant. If the official
assistant's embed_tokens / lm_head are IDENTICAL to the target's (i.e. the
assistant just reuses the target's embedding table and never trained them), then
we should FREEZE them during our continued training too — otherwise we'd drift
away from a layer the official pipeline deliberately kept fixed.

Prints, for embed_tokens and lm_head (and any tied relationship):
  - shapes
  - max abs diff / cosine vs the target's corresponding weight
  - whether assistant lm_head is tied to its own embed_tokens

USAGE:
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/compare_assistant_target_embed.py \
      --official $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
      --target   $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from safetensors import safe_open


def load_weight_map(path):
    """Return {tensor_name: file} across all safetensors shards in a dir."""
    import glob
    import json
    import os

    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return {k: os.path.join(path, v)
                    for k, v in json.load(f)["weight_map"].items()}
    # single shard
    files = glob.glob(os.path.join(path, "*.safetensors"))
    wm = {}
    for fp in files:
        with safe_open(fp, framework="pt") as f:
            for k in f.keys():
                wm[k] = fp
    return wm


def get_tensor(wm, name):
    if name not in wm:
        return None
    with safe_open(wm[name], framework="pt") as f:
        return f.get_tensor(name)


def find_key(wm, *substrings):
    for k in wm:
        if all(s in k for s in substrings):
            return k
    return None


def compare(a, b, label):
    if a is None or b is None:
        print(f"  [{label}] MISSING (a={a is not None}, b={b is not None})")
        return
    a = a.float()
    b = b.float()
    if a.shape != b.shape:
        print(f"  [{label}] SHAPE DIFF a={tuple(a.shape)} b={tuple(b.shape)} "
              f"-> independent layers, not comparable (do NOT freeze on this basis)")
        return
    maxdiff = (a - b).abs().max().item()
    cos = F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=-1).item()
    same = "IDENTICAL" if maxdiff < 1e-6 else "DIFFERENT"
    print(f"  [{label}] shape={tuple(a.shape)} maxdiff={maxdiff:.3e} "
          f"cos={cos:.6f} -> {same}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--target", required=True)
    args = ap.parse_args()

    wm_a = load_weight_map(args.official)
    wm_t = load_weight_map(args.target)

    print("=== ASSISTANT tensor names (embed/lm_head/norm) ===")
    for k in sorted(wm_a):
        if any(s in k for s in ("embed", "lm_head", "norm", "projection")):
            print(f"  A: {k}")
    print("=== TARGET tensor names (embed/lm_head) ===")
    for k in sorted(wm_t):
        if any(s in k for s in ("embed", "lm_head")):
            print(f"  T: {k}")

    a_embed_k = find_key(wm_a, "embed_tokens", "weight")
    a_lm_k = find_key(wm_a, "lm_head", "weight")
    t_embed_k = find_key(wm_t, "embed_tokens", "weight")
    t_lm_k = find_key(wm_t, "lm_head", "weight")

    a_embed = get_tensor(wm_a, a_embed_k) if a_embed_k else None
    a_lm = get_tensor(wm_a, a_lm_k) if a_lm_k else None
    t_embed = get_tensor(wm_t, t_embed_k) if t_embed_k else None
    t_lm = get_tensor(wm_t, t_lm_k) if t_lm_k else None

    print("\n=== assistant embed vs target embed ===")
    compare(a_embed, t_embed, "embed_tokens: assistant vs target")
    print("=== assistant lm_head vs target lm_head ===")
    compare(a_lm, t_lm, "lm_head: assistant vs target")
    if t_lm is None and t_embed is not None:
        print("  (target has no separate lm_head -> tied to target embed)")
        compare(a_lm, t_embed, "assistant lm_head vs target embed (tied)")

    print("\n=== is assistant lm_head tied to assistant embed? ===")
    compare(a_lm, a_embed, "assistant lm_head vs assistant embed")

    print("\n" + "=" * 60)
    print("READING: if 'assistant vs target' is IDENTICAL for embed/lm_head,")
    print("the official assistant reused the target's table unchanged -> FREEZE")
    print("those layers in continued training. If DIFFERENT, they were trained")
    print("-> keep them trainable.")
    print("=" * 60)


if __name__ == "__main__":
    main()
