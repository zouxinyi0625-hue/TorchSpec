#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
"""
Convert a TorchSpec DCP training checkpoint (iter_XXXXXXX/model/) into a
standard HuggingFace Gemma4AssistantForCausalLM directory that vLLM / HF
`from_pretrained` can load directly.

WHY: training saves the draft via torch.distributed.checkpoint (DCP) as
sharded files under iter_XXXXXXX/model/. The state_dict keys are nested as
    model_state -> model -> draft_model.assistant.<hf key>
This tool loads the DCP shards, strips the "draft_model.assistant." prefix,
instantiates a fresh Gemma4AssistantForCausalLM from the assistant config,
loads the weights (strict), and writes config.json + safetensors.

USAGE:
    python tools/gemma4_mtp/convert_dcp_to_hf.py \
        --ckpt   outputs/gemma4-mtp/exp1_layer1_delta/checkpoints/iter_0002269 \
        --assistant-config /tmp/models/gemma4/assistant \
        --out    outputs/gemma4-mtp/exp1_layer1_delta/hf_iter_0002269

Then in vLLM point the draft/speculative model at --out.

NOTES:
  * --assistant-config is the ORIGINAL assistant dir (has the correct
    Gemma4AssistantConfig: layer_types, rope, hidden_size_per_layer_input...).
    We only take its config; weights come from the DCP checkpoint.
  * Runs single-process on CPU. DCP can consolidate sharded -> full on load
    with no distributed init required.
"""
import argparse
import os

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint import FileSystemReader


PREFIX = "draft_model.assistant."


def load_dcp_model_state(model_dir: str) -> dict:
    """Load the DCP-sharded model state into a flat {key: tensor} dict on CPU.

    The saved structure is {"model_state": {"model": <flat state_dict>}}.
    We provide an empty template and let DCP fill tensors in-place.
    """
    reader = FileSystemReader(model_dir)
    # Read metadata to discover the exact keys stored in the checkpoint.
    metadata = reader.read_metadata()
    stored_keys = list(metadata.state_dict_metadata.keys())

    # Build an empty state_dict template matching the stored (possibly nested)
    # keys so dcp.load knows what to materialize. DCP flattens nested dicts
    # with "." — keys look like "model_state.model.draft_model.assistant.xxx".
    template = {k: torch.empty(0) for k in stored_keys}
    dcp.load(template, checkpoint_id=model_dir)
    return template, stored_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="iter_XXXXXXX dir (contains model/ subdir), or the model/ dir itself")
    ap.add_argument("--assistant-config", required=True,
                    help="Original assistant dir with the correct Gemma4AssistantConfig")
    ap.add_argument("--out", required=True, help="Output HF directory")
    ap.add_argument("--safe-serialization", action="store_true", default=True,
                    help="Write safetensors (default) vs pytorch_model.bin")
    args = ap.parse_args()

    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM

    model_dir = args.ckpt
    if os.path.isdir(os.path.join(model_dir, "model")):
        model_dir = os.path.join(model_dir, "model")
    print(f"[1/4] Loading DCP shards from {model_dir}")

    flat, stored_keys = load_dcp_model_state(model_dir)
    print(f"      loaded {len(flat)} tensors; sample keys:")
    for k in stored_keys[:3]:
        print(f"        {k}  {tuple(flat[k].shape)}")

    # Strip the DCP nesting + module prefix down to raw HF assistant keys.
    # Stored key form: "model_state.model.draft_model.assistant.<hf key>"
    print("[2/4] Remapping keys -> HF Gemma4Assistant namespace")
    hf_state = {}
    dropped = []
    for k, v in flat.items():
        kk = k
        for pre in ("model_state.", "model."):
            if kk.startswith(pre):
                kk = kk[len(pre):]
        if kk.startswith(PREFIX):
            hf_state[kk[len(PREFIX):]] = v
        else:
            dropped.append(kk)
    print(f"      kept {len(hf_state)} assistant weights, dropped {len(dropped)} non-assistant keys")
    if dropped:
        print(f"      (dropped e.g.: {dropped[:5]})")

    print(f"[3/4] Building Gemma4AssistantForCausalLM from {args.assistant_config}")
    cfg = Gemma4AssistantConfig.from_pretrained(args.assistant_config)
    model = Gemma4AssistantForCausalLM(cfg)

    missing, unexpected = model.load_state_dict(hf_state, strict=False)
    # Tied lm_head may legitimately be "missing" (shares embed_tokens).
    real_missing = [m for m in missing if "lm_head" not in m]
    if real_missing:
        raise SystemExit(f"ERROR: missing weights after load: {real_missing[:10]}")
    if unexpected:
        print(f"      WARNING unexpected keys ignored: {unexpected[:10]}")
    model = model.to(torch.bfloat16).eval()

    print(f"[4/4] Saving HF checkpoint to {args.out}")
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=args.safe_serialization)
    cfg.save_pretrained(args.out)
    # Carry the tokenizer over so vLLM can load a self-contained dir.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.assistant_config)
        tok.save_pretrained(args.out)
        print("      tokenizer copied")
    except Exception as e:
        print(f"      tokenizer copy skipped ({e}); point vLLM tokenizer at {args.assistant_config}")

    print(f"DONE. Load in vLLM with model path: {args.out}")


if __name__ == "__main__":
    main()
