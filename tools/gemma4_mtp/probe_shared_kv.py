#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
"""
Capture the EXACT shared_kv_states contract the Gemma4 assistant consumes.

The assistant forward takes `shared_kv_states: dict[str, tuple[K, V]]` — one
entry per layer_type (full_attention / sliding_attention), being the KV of the
target's LAST layer of that type. To stream this over Mooncake (option A) we
need the precise dict keys, tensor shapes and dtypes.

This probe runs the TARGET, hooks the attention modules to grab the produced
K/V for the last layer of each layer_type, and prints the shapes. It also runs
the ASSISTANT once with those states to confirm the shapes are accepted.

Run on the server (GPU):

    python tools/gemma4_mtp/probe_shared_kv.py

Paste stdout back.
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="/tmp/models/gemma4/text_only")
    ap.add_argument("--assistant", default="/tmp/models/gemma4/assistant")
    ap.add_argument("--seq-text", default="The quick brown fox jumps over the lazy dog")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tcfg = AutoConfig.from_pretrained(args.target, trust_remote_code=True).text_config
    layer_types = tcfg.layer_types
    print(f"[TARGET] layer_types (len={len(layer_types)}): "
          f"unique={sorted(set(layer_types))}")
    # last layer index for each type
    last_of_type = {}
    for i, lt in enumerate(layer_types):
        last_of_type[lt] = i
    print(f"[TARGET] last layer index per type: {last_of_type}")

    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, trust_remote_code=True
    ).eval().cuda()
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    ids = tok(args.seq_text, return_tensors="pt").input_ids.cuda()

    # --- locate target decoder layers + their attention submodule ---
    lm = getattr(target, "model", target)
    lang = getattr(lm, "language_model", lm)
    layers = lang.layers
    print(f"[TARGET] num decoder layers = {len(layers)}")

    captured = {}

    def make_hook(idx, lt):
        def hook(module, inp, out):
            # Grab whatever K/V-like tensors the attention exposes. We snapshot
            # the module's most recent q/k/v by re-projecting from the input
            # hidden states is fragile; instead inspect `out` and module attrs.
            captured[(idx, lt)] = {
                "out_type": type(out).__name__,
                "out_shapes": [tuple(o.shape) for o in out
                               if isinstance(o, torch.Tensor)]
                if isinstance(out, (tuple, list)) else tuple(out.shape),
            }
        return hook

    handles = []
    for lt, idx in last_of_type.items():
        attn = getattr(layers[idx], "self_attn", None)
        if attn is not None:
            handles.append(attn.register_forward_hook(make_hook(idx, lt)))

    with torch.no_grad():
        # Ask HF to return shared_kv_states if the model supports it.
        try:
            out = target.model(input_ids=ids, use_cache=True,
                               return_shared_kv_states=True)
            sks = getattr(out, "shared_kv_states", None)
        except TypeError:
            out = target(input_ids=ids, use_cache=True)
            sks = None

    for h in handles:
        h.remove()

    print("\n[ATTN HOOK CAPTURES] (last layer of each type)")
    for (idx, lt), info in captured.items():
        print(f"  layer {idx:>3} [{lt}]: {info}")

    print(f"\n[SHARED_KV_STATES from target.model output] present={sks is not None}")
    if sks is not None:
        for k, v in sks.items():
            if isinstance(v, (tuple, list)):
                shapes = [tuple(t.shape) if isinstance(t, torch.Tensor) else t
                          for t in v]
                dtypes = [t.dtype if isinstance(t, torch.Tensor) else None
                          for t in v]
                print(f"  '{k}': tuple(len={len(v)}) shapes={shapes} dtypes={dtypes}")
            elif isinstance(v, torch.Tensor):
                print(f"  '{k}': tensor {tuple(v.shape)} {v.dtype}")
            else:
                print(f"  '{k}': {type(v).__name__}")

    # --- confirm the assistant accepts these states ---
    if sks is not None:
        print("\n[ASSISTANT] feeding shared_kv_states into assistant.forward ...")
        asst = AutoModelForCausalLM.from_pretrained(
            args.assistant, dtype=torch.bfloat16, trust_remote_code=True
        ).eval().cuda()
        target_last_hidden = out.last_hidden_state  # (B, T, 2816)
        # concat[target_embed(token), prev_hidden] -> here prev_hidden = target hidden
        emb = target.get_input_embeddings()(ids)  # (B, T, 2816) already scaled
        inputs_embeds = torch.cat([emb, target_last_hidden], dim=-1)  # (B, T, 5632)
        print(f"  inputs_embeds shape = {tuple(inputs_embeds.shape)} (expect ...5632)")
        with torch.no_grad():
            a_out = asst(inputs_embeds=inputs_embeds, shared_kv_states=sks)
        print(f"  assistant logits    = {tuple(a_out.logits.shape)}")
        print(f"  assistant last_hs   = {tuple(a_out.last_hidden_state.shape)} "
              f"(post_projection -> expect ...2816)")

    print("\n# DONE. Paste everything above back.")


if __name__ == "__main__":
    main()
