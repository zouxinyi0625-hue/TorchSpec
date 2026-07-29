#!/usr/bin/env python
"""Smoke test: build Gemma4DSparkDraftModel from a gemma4 target config and run a
forward pass with dummy inputs. Verifies config derivation + Model-level wiring +
dual-source-KV block forward (no crash, sane shapes). Run on the server (needs
torch/transformers). NOT a numerical parity check — that's the next step (dump
vllm dspark forward and compare argmax).

Usage:
  python tools/gemma4_dspark/smoke_backbone.py --target $AZURE_..._text_only
"""

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="gemma4 target model path")
    ap.add_argument("--num-draft-layers", type=int, default=1)
    ap.add_argument("--num-target-layers", type=int, default=5)
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--num-anchors", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoConfig

    from torchspec.models.draft.dspark import DSparkConfig
    from torchspec.models.draft.dspark_gemma4_backbone import Gemma4DSparkDraftModel
    from torchspec.models.draft.dspark_gemma4_config import (
        build_gemma4_dspark_draft_config_dict,
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    print("loading target config ...")
    target_config = AutoConfig.from_pretrained(args.target, trust_remote_code=True)

    cfg_dict = build_gemma4_dspark_draft_config_dict(
        target_config,
        num_draft_layers=args.num_draft_layers,
        num_target_layers=args.num_target_layers,
        block_size=args.block_size,
        num_anchors=args.num_anchors,
    )
    print("derived draft config keys:", sorted(cfg_dict.keys()))
    print("  hidden_size          =", cfg_dict["hidden_size"])
    print("  num_hidden_layers    =", cfg_dict["num_hidden_layers"])
    print("  num_attention_heads  =", cfg_dict["num_attention_heads"])
    print("  head_dim / global    =", cfg_dict.get("head_dim"), cfg_dict.get("global_head_dim"))
    print("  num_key_value_heads  =", cfg_dict["num_key_value_heads"])
    print("  attention_k_eq_v     =", cfg_dict.get("attention_k_eq_v"))
    print("  num_global_kv_heads  =", cfg_dict.get("num_global_key_value_heads"))
    print("  target_layer_ids     =", cfg_dict["target_layer_ids"])
    print("  target_hidden_size   =", cfg_dict["target_hidden_size"])
    print("  markov_rank          =", cfg_dict["markov_rank"])
    print("  enable_confidence    =", cfg_dict["enable_confidence_head"])

    config = DSparkConfig(**cfg_dict)
    model = Gemma4DSparkDraftModel(config).to(dev).to(dtype).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nGemma4DSparkDraftModel built: {n_params/1e6:.1f}M params, dtype={dtype}")
    print("  has markov_head     =", model.markov_head is not None)
    print("  has confidence_head =", model.confidence_head is not None)

    # ---- dummy forward (no block mask -> bidirectional SDPA fallback) ----
    B = 1
    hs = config.hidden_size
    ths = config.target_hidden_size
    nb = args.num_anchors
    bs = args.block_size
    draft_len = nb * bs
    ctx_len = args.seq_len

    hidden_states_list = [
        torch.randn(B, ctx_len, ths, device=dev, dtype=dtype)
        for _ in range(args.num_target_layers)
    ]
    context_feature = model.extract_context_feature(hidden_states_list)
    print("\ncontext_feature:", tuple(context_feature.shape), "(expect [B, ctx_len, hidden])")

    noise_embedding = torch.randn(B, draft_len, hs, device=dev, dtype=dtype)
    draft_position_ids = torch.arange(draft_len, device=dev).unsqueeze(0)
    context_position_ids = torch.arange(ctx_len, device=dev).unsqueeze(0)

    with torch.no_grad():
        out = model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=None,
            noise_embedding=noise_embedding,
        )
    print("draft forward out:", tuple(out.shape), "(expect [B, draft_len, hidden])")
    assert out.shape == (B, draft_len, hs), "unexpected output shape"
    assert torch.isfinite(out).all(), "non-finite output"
    print("\n=> SMOKE OK: gemma4 dspark backbone builds + forwards with sane shapes.")


if __name__ == "__main__":
    main()
