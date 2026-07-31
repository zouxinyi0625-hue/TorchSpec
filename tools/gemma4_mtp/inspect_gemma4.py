#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
"""
Inspect a Gemma4 *target* model and (optionally) its MTP / assistant draft model,
so we can implement Gemma4 MTP training in TorchSpec against the REAL structure.

Run on the server where the weights live:

    python tools/gemma4_mtp/inspect_gemma4.py \
        --target /path/to/gemma4-target \
        --assistant /path/to/gemma4-mtp-or-assistant   # optional

Paste the full stdout back. It does NOT load weights to GPU; it reads config +
safetensors headers (metadata only) so it is fast and low-memory.
"""
import argparse
import glob
import json
import os
from collections import OrderedDict


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def dump_config(model_dir, tag):
    print(f"\n{'=' * 70}\n[{tag}] config @ {model_dir}\n{'=' * 70}")
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"  !! no config.json in {model_dir}")
        return {}
    cfg = _load_json(cfg_path)
    # Print the whole thing — small and we want every field.
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    # Highlight the fields that matter for MTP wiring.
    keys = [
        "model_type", "architectures", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "vocab_size", "rms_norm_eps", "rope_theta",
        "max_position_embeddings", "tie_word_embeddings", "torch_dtype",
        "query_pre_attn_scalar", "sliding_window", "attn_logit_softcapping",
        "final_logit_softcapping", "num_nextn_predict_layers",  # MTP hints
        "mtp_num_layers", "n_future_tokens", "embedding_multiplier_scale",
    ]
    print(f"\n[{tag}] KEY FIELDS:")
    for k in keys:
        if k in cfg:
            print(f"  {k:32s} = {cfg[k]}")
    return cfg


def dump_weight_index(model_dir, tag):
    """List every tensor name + shape + dtype from safetensors headers (no load)."""
    from safetensors import safe_open

    print(f"\n{'-' * 70}\n[{tag}] WEIGHT TENSORS (name : shape : dtype)\n{'-' * 70}")
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not st_files:
        print(f"  !! no .safetensors in {model_dir}")
        return
    tensors = OrderedDict()
    for stf in st_files:
        with safe_open(stf, framework="pt") as f:
            for key in f.keys():
                sl = f.get_slice(key)
                tensors[key] = (tuple(sl.get_shape()), sl.get_dtype())
    # Group by top-level prefix to keep output scannable.
    groups = OrderedDict()
    for name, (shape, dtype) in tensors.items():
        top = ".".join(name.split(".")[:2])
        groups.setdefault(top, []).append((name, shape, dtype))
    for top, items in groups.items():
        print(f"\n  # {top}  ({len(items)} tensors)")
        for name, shape, dtype in items[:12]:  # cap per group; layers repeat
            print(f"    {name:60s} {str(shape):22s} {dtype}")
        if len(items) > 12:
            print(f"    ... (+{len(items) - 12} more with same pattern)")
    print(f"\n[{tag}] total tensors: {len(tensors)}")


def check_embed_scale(model_dir, tag):
    """
    Gemma applies embed_scale = sqrt(hidden_size) inside get_input_embeddings.
    Confirm hidden_size so we know the scale value, and flag the double-scaling trap.
    """
    cfg = _load_json(os.path.join(model_dir, "config.json"))
    h = cfg.get("hidden_size")
    if h:
        import math
        print(f"\n[{tag}] embed_scale (Gemma) = sqrt({h}) = {math.sqrt(h):.6f}")
        print(f"  -> MTP input concat MUST use RAW target_embed(token) (already scaled);")
        print(f"     do NOT add a manual normalizer (double-scaling kills accept rate).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="/tmp/models/gemma4/text_only",
                    help="Gemma4 target model dir")
    ap.add_argument("--assistant", default="/tmp/models/gemma4/assistant",
                    help="MTP/assistant draft model dir")
    args = ap.parse_args()

    print("#" * 70)
    print("# TorchSpec Gemma4 MTP — model inspection")
    print("#" * 70)

    dump_config(args.target, "TARGET")
    check_embed_scale(args.target, "TARGET")
    dump_weight_index(args.target, "TARGET")

    if args.assistant:
        dump_config(args.assistant, "ASSISTANT/MTP")
        check_embed_scale(args.assistant, "ASSISTANT/MTP")
        dump_weight_index(args.assistant, "ASSISTANT/MTP")

    print("\n# DONE. Paste everything above back.")


if __name__ == "__main__":
    main()
