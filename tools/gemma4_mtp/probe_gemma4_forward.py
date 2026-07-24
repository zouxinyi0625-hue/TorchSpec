#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
"""
Probe the Gemma4 target forward pass to capture the exact hidden-state contract
TorchSpec's MTP draft must consume: which layers produce aux hidden states, the
shape/dtype of embeddings, whether embed_scale is baked in, and the final-norm
behaviour.

Run on the server (needs GPU + weights; loads the model):

    python tools/gemma4_mtp/probe_gemma4_forward.py \
        --target /path/to/gemma4-target \
        --aux-layers 2 8 16          # optional guess; else auto-picks low/mid/high

Paste stdout back. This confirms the runtime behaviour the config alone can't
show (e.g. whether get_input_embeddings already multiplies by sqrt(hidden)).
"""
import argparse
import math

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="/tmp/models/gemma4/text_only")
    ap.add_argument("--aux-layers", type=int, nargs="*", default=None)
    ap.add_argument("--seq", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(args.target, trust_remote_code=True)
    hidden = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    print(f"model_type={cfg.model_type} hidden={hidden} layers={n_layers} "
          f"vocab={cfg.vocab_size}")

    model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, trust_remote_code=True,
        output_hidden_states=True,
    ).eval().cuda()

    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    ids = tok("The quick brown fox jumps", return_tensors="pt").input_ids.cuda()

    # ---- embed_scale check: does get_input_embeddings bake in sqrt(hidden)? ----
    emb_layer = model.get_input_embeddings()
    with torch.no_grad():
        raw_lookup = emb_layer.weight[ids[0]]            # direct weight lookup
        via_call = emb_layer(ids)[0]                     # through the module
    ratio = (via_call.norm() / raw_lookup.norm()).item()
    print(f"\n[EMBED] ||embed(ids)|| / ||weight[ids]|| = {ratio:.4f} "
          f"(sqrt(hidden)={math.sqrt(hidden):.4f})")
    print("  -> if ratio≈1: scale baked into WEIGHTS; if ratio≈sqrt(hidden): "
          "scale applied in forward. Either way MTP must reuse embed(ids) AS-IS.")

    # ---- aux hidden states ----
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states  # tuple: len = n_layers + 1 (embeddings + each layer)
    print(f"\n[HIDDEN] output_hidden_states tuple len = {len(hs)} "
          f"(expect n_layers+1 = {n_layers + 1})")
    for i, h in enumerate(hs):
        if i in (0, 1, n_layers // 2, n_layers - 1, n_layers):
            print(f"  hs[{i:>3}] shape={tuple(h.shape)} dtype={h.dtype} "
                  f"mean_norm={h.float().norm(dim=-1).mean().item():.3f}")

    aux = args.aux_layers or [max(1, n_layers // 4), n_layers // 2,
                              max(1, n_layers - n_layers // 4)]
    print(f"\n[AUX] suggested aux_hidden_state_layer_ids = {aux}")
    print(f"  concat dim would be {len(aux)} * {hidden} = {len(aux) * hidden}")

    # ---- final norm ----
    fn = getattr(model.model, "norm", None)
    print(f"\n[FINAL_NORM] type={type(fn).__name__ if fn else None}")
    if fn is not None and hasattr(fn, "weight"):
        print(f"  norm.weight shape={tuple(fn.weight.shape)} "
              f"eps={getattr(fn, 'variance_epsilon', getattr(fn, 'eps', '?'))}")
        # Gemma RMSNorm uses (1 + weight) — flag it.
        print(f"  weight mean={fn.weight.float().mean().item():.4f} "
              f"(Gemma RMSNorm scales by (1+weight); check the impl!)")

    print("\n# DONE. Paste everything above back.")


if __name__ == "__main__":
    main()
