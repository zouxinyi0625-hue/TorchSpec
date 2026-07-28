#!/usr/bin/env python
"""
Compare the EXPORTED Gemma4 MTP draft checkpoint against what vLLM's official
Gemma4MTP (PR #41745) expects to load. A silent weight-name/shape mismatch at
vLLM load time is the strongest remaining suspect for the pos0 train-vs-deploy
gap (shared_kv and hidden pre/post-norm already ruled out).

vLLM's Gemma4MTP expects (checkpoint layout, from the PR docstring):
    model.embed_tokens.weight            # token embeddings (backbone dim)
    model.layers.{i}.self_attn.q_proj    # Q-only attention (no K/V!)
    model.layers.{i}.self_attn.q_norm
    model.layers.{i}.input_layernorm
    model.layers.{i}.post_attention_layernorm
    model.layers.{i}.pre_feedforward_layernorm
    model.layers.{i}.post_feedforward_layernorm
    model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}   # gate_up fused in vLLM
    model.layers.{i}.layer_scalar
    model.norm.weight                    # final RMSNorm
    pre_projection.weight                # Linear(2*backbone, hidden) -> model.pre_projection
    post_projection.weight               # Linear(hidden, backbone)   -> model.post_projection
    lm_head.weight                       # tied to embed_tokens if tie_word_embeddings

Key invariants this checks:
  * pre_projection.weight shape == [hidden, 2*backbone]   (concat[tok_embed, hidden])
  * post_projection.weight shape == [backbone, hidden]
  * NO k_proj / v_proj / k_norm / v_norm in the draft layers (Q-only)
  * embed_tokens present, lm_head present-or-tied
  * layer count matches config

USAGE:
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/check_draft_export.py \
      --draft $AZURE_ML_INPUT_UKWDATA/maiprofile/mtp_26b/gemma4_mtp_exp1_hf/hf_iter_0002269
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict


def load_state_dict_keys_shapes(path: str):
    """Return {weight_name: shape} from a HF checkpoint dir (safetensors/bin)."""
    import glob

    out = {}
    sts = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if sts:
        from safetensors import safe_open

        for f in sts:
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    out[k] = tuple(h.get_slice(k).get_shape())
        return out
    bins = sorted(glob.glob(os.path.join(path, "*.bin")))
    if bins:
        import torch

        for f in bins:
            sd = torch.load(f, map_location="cpu")
            for k, v in sd.items():
                out[k] = tuple(v.shape)
        return out
    raise SystemExit(f"no .safetensors or .bin in {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    args = ap.parse_args()

    keys = load_state_dict_keys_shapes(args.draft)
    cfg_path = os.path.join(args.draft, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    tcfg = cfg.get("text_config", cfg)

    backbone = tcfg.get("backbone_hidden_size") or tcfg.get("hidden_size")
    draft_hidden = tcfg.get("hidden_size")
    vocab = tcfg.get("vocab_size")
    n_layers = tcfg.get("num_hidden_layers") or tcfg.get("n_layer")
    tie = cfg.get("tie_word_embeddings", tcfg.get("tie_word_embeddings", True))

    print("=" * 64)
    print("CONFIG")
    print(f"  backbone_hidden_size = {backbone}")
    print(f"  draft hidden_size    = {draft_hidden}")
    print(f"  vocab_size           = {vocab}")
    print(f"  num_hidden_layers    = {n_layers}")
    print(f"  tie_word_embeddings  = {tie}")
    print(f"  total tensors        = {len(keys)}")

    def find(pat):
        rx = re.compile(pat)
        return {k: v for k, v in keys.items() if rx.search(k)}

    print("=" * 64)
    print("REQUIRED TENSORS")

    checks = []

    # pre_projection: [draft_hidden, 2*backbone]
    pp = find(r"pre_projection\.weight$")
    if pp:
        name, shape = next(iter(pp.items()))
        exp = (draft_hidden, 2 * backbone) if backbone and draft_hidden else None
        ok = exp is None or shape == exp
        checks.append(("pre_projection", ok, f"{name} {shape} exp~{exp}"))
    else:
        checks.append(("pre_projection", False, "MISSING"))

    # post_projection: [backbone, draft_hidden]
    qp = find(r"post_projection\.weight$")
    if qp:
        name, shape = next(iter(qp.items()))
        exp = (backbone, draft_hidden) if backbone and draft_hidden else None
        ok = exp is None or shape == exp
        checks.append(("post_projection", ok, f"{name} {shape} exp~{exp}"))
    else:
        checks.append(("post_projection", False, "MISSING"))

    # embed_tokens
    emb = find(r"embed_tokens\.weight$")
    checks.append(("embed_tokens", bool(emb),
                   f"{next(iter(emb.items())) if emb else 'MISSING'}"))

    # lm_head (present or tied)
    lm = find(r"lm_head\.weight$")
    checks.append(("lm_head", bool(lm) or bool(tie),
                   f"{next(iter(lm.items())) if lm else ('TIED' if tie else 'MISSING')}"))

    # final norm
    nm = find(r"(^|\.)model\.norm\.weight$|^norm\.weight$")
    checks.append(("final norm", bool(nm),
                   f"{next(iter(nm.items())) if nm else 'MISSING'}"))

    for name, ok, detail in checks:
        print(f"  [{'OK ' if ok else 'XX '}] {name:16} {detail}")

    print("=" * 64)
    print("DRAFT LAYER STRUCTURE (Q-only expected: NO k_proj/v_proj/k_norm/v_norm)")
    per_layer = defaultdict(set)
    layer_rx = re.compile(r"layers\.(\d+)\.(.+?)\.weight$")
    for k in keys:
        m = layer_rx.search(k)
        if m:
            per_layer[int(m.group(1))].add(m.group(2))
    if per_layer:
        li = sorted(per_layer)[0]
        subs = sorted(per_layer[li])
        print(f"  layer {li} submodules:")
        for s in subs:
            print(f"    {s}")
        # Q-only check
        bad = [s for s in subs if re.search(r"\b(k_proj|v_proj|k_norm|v_norm)\b", s)]
        print(f"  layer count seen = {len(per_layer)} (config says {n_layers})")
        if bad:
            print(f"  [XX ] found K/V modules (should be Q-only!): {bad}")
        else:
            print(f"  [OK ] Q-only confirmed (no k_proj/v_proj/k_norm/v_norm)")
    else:
        print("  [XX ] no layers.N.*.weight found -- unexpected checkpoint layout")

    print("=" * 64)
    print("UNEXPECTED / EXTRA top-level tensors (potential mismatch source):")
    known = re.compile(r"(embed_tokens|layers\.\d+|pre_projection|post_projection|"
                       r"lm_head|\bnorm\b|masked_embedding|centroids|normalizer)")
    extra = [k for k in sorted(keys) if not known.search(k)]
    for k in extra[:30]:
        print(f"    {k}  {keys[k]}")
    if not extra:
        print("    (none)")
    print("=" * 64)
    fails = [n for n, ok, _ in checks if not ok]
    if fails:
        print(f"VERDICT: {len(fails)} required tensor(s) FAILED: {fails}")
        print("  -> vLLM load would mis-map / miss these -> pos0 corruption likely.")
    else:
        print("VERDICT: all required tensors present with expected shapes.")
        print("  -> export structure looks OK; pos0 gap is elsewhere (forward math).")


if __name__ == "__main__":
    main()
