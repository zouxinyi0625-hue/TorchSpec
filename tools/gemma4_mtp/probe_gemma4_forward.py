#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
"""
Probe the Gemma4 target + assistant(MTP) runtime contract for TorchSpec MTP.

Gemma4Config is NESTED: real fields live under cfg.text_config, not top level.
This version handles that, confirms embed_scale behaviour, dumps aux hidden
shapes, and (critically) prints the assistant's module tree + forward SOURCE so
we can copy the pre_projection / post_projection / recurrence contract exactly.

Run on the server (needs GPU + weights):

    python tools/gemma4_mtp/probe_gemma4_forward.py \
        --target    /tmp/models/gemma4/text_only \
        --assistant /tmp/models/gemma4/assistant

Paste stdout back.
"""
import argparse
import inspect
import math

import torch


def _text_cfg(cfg):
    """Gemma4 keeps the real transformer config under .text_config."""
    return getattr(cfg, "text_config", None) or cfg


def probe_target(path, seq_text):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 70 + f"\n[TARGET] {path}\n" + "=" * 70)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    tcfg = _text_cfg(cfg)
    hidden = tcfg.hidden_size
    n_layers = tcfg.num_hidden_layers
    print(f"model_type={cfg.model_type} hidden={hidden} layers={n_layers} "
          f"vocab={tcfg.vocab_size} head_dim={getattr(tcfg, 'head_dim', '?')}")

    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, trust_remote_code=True,
    ).eval().cuda()

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    ids = tok(seq_text, return_tensors="pt").input_ids.cuda()

    # ---- embed_scale: is sqrt(hidden) baked into weights or applied in forward? ----
    emb = model.get_input_embeddings()
    with torch.no_grad():
        raw = emb.weight[ids[0]]
        called = emb(ids)[0]
    ratio = (called.norm() / raw.norm()).item()
    print(f"\n[EMBED] ||embed(ids)|| / ||weight[ids]|| = {ratio:.4f} "
          f"(sqrt(hidden)={math.sqrt(hidden):.4f})")
    print("  ratio~1 -> scale in WEIGHTS ; ratio~sqrt(hidden) -> scale in FORWARD.")
    print("  MTP must reuse embed(token) AS-IS (no extra normalizer).")

    # ---- aux hidden states ----
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states
    print(f"\n[HIDDEN] tuple len = {len(hs)} (n_layers+1 = {n_layers + 1})")
    for i, h in enumerate(hs):
        if i in (0, 1, n_layers // 2, n_layers - 1, n_layers):
            print(f"  hs[{i:>3}] shape={tuple(h.shape)} dtype={h.dtype} "
                  f"mean_norm={h.float().norm(dim=-1).mean().item():.3f}")

    fn = getattr(model.model, "norm", None) or getattr(
        getattr(model.model, "language_model", model.model), "norm", None)
    print(f"\n[FINAL_NORM] type={type(fn).__name__ if fn else None}")
    if fn is not None and hasattr(fn, "weight"):
        print(f"  norm.weight shape={tuple(fn.weight.shape)} "
              f"mean={fn.weight.float().mean().item():.4f} "
              f"(Gemma RMSNorm uses (1+weight))")
    return hidden


def probe_assistant(path, target_hidden):
    from transformers import AutoConfig, AutoModelForCausalLM

    print("\n" + "=" * 70 + f"\n[ASSISTANT/MTP] {path}\n" + "=" * 70)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    tcfg = _text_cfg(cfg)
    print(f"model_type={cfg.model_type} draft_hidden={tcfg.hidden_size} "
          f"layers={tcfg.num_hidden_layers} "
          f"backbone_hidden={getattr(cfg, 'backbone_hidden_size', '?')}")

    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, trust_remote_code=True,
    ).eval()

    # ---- module tree (top 2 levels) ----
    print("\n[MODULE TREE]")
    for name, mod in model.named_modules():
        depth = name.count(".")
        if name and depth <= 1 and "layers." not in name.replace("layers.0", ""):
            if ".layers." in name and not name.endswith("layers.0"):
                continue
            print(f"  {name:40s} {type(mod).__name__}")

    # ---- forward source of the top model + inner model ----
    for obj, tag in [(model, "TOP.forward"),
                     (getattr(model, "model", None), "INNER.forward")]:
        if obj is None:
            continue
        try:
            src = inspect.getsource(obj.forward)
            print(f"\n[SOURCE {tag}]\n{src}")
        except (OSError, TypeError) as e:
            print(f"\n[SOURCE {tag}] unavailable: {e}")

    # ---- where do pre/post projection sit + shapes ----
    print("\n[PROJECTIONS]")
    for n, p in model.named_parameters():
        if "projection" in n or n.endswith("embed_tokens.weight") or n == "lm_head.weight":
            print(f"  {n:40s} {tuple(p.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="/tmp/models/gemma4/text_only")
    ap.add_argument("--assistant", default="/tmp/models/gemma4/assistant")
    ap.add_argument("--seq-text", default="The quick brown fox jumps")
    ap.add_argument("--skip-target", action="store_true")
    args = ap.parse_args()

    th = 2816
    if not args.skip_target:
        th = probe_target(args.target, args.seq_text)
    probe_assistant(args.assistant, th)
    print("\n# DONE. Paste everything above back.")


if __name__ == "__main__":
    main()
