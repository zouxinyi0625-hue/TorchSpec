#!/usr/bin/env python
"""
STRICT single-prompt pos0 diff: feed the training target the EXACT token_ids
vLLM fed the draft, then compare hidden per-position (no tokenizer, no prompt
mismatch).

Run vLLM bench with NUM_PROMPTS=1 and VLLM_MTP_DUMP_DIR set so the dump is a
SINGLE prompt (not a mixed batch). Then this script:
  1. loads vLLM's dumped target_token_ids + target_hidden_states
  2. runs the HF training target on those SAME token_ids
  3. compares hidden per-position (cosine + L2 ratio), aligned token-by-token

If per-position hidden matches (cos~1, ratio~1), vLLM feeds the draft the same
hidden the training target produced -> hidden is NOT the pos0 gap; look at
shared_kv next. If they diverge, that's the pos0 root cause.

USAGE (server, 1 GPU), TorchSpec root:
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/diff_pos0_strict.py \
      --target /tmp/models/gemma4/text_only \
      --vllm-dump /tmp/mtp_dump/vllm_pos0_inputs.pt
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def find_backbone(m):
    import torch.nn as nn
    cands, seen = [m], set()
    while cands:
        mod = cands.pop(0)
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        if (hasattr(mod, "norm") and isinstance(getattr(mod, "norm"), nn.Module)
                and hasattr(mod, "layers")):
            return mod
        for _n, c in mod.named_children():
            cands.append(c)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--vllm-dump", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    v = torch.load(args.vllm_dump, map_location="cpu")
    vtok = v["target_token_ids"].reshape(-1).to(torch.long)     # (T,)
    vhid = v["target_hidden_states"].float()                    # (T, 2816)
    T = vtok.shape[0]
    print(f"vLLM dump: {T} tokens, hidden {tuple(vhid.shape)}")
    if T > 4096:
        print(f"WARNING: {T} tokens looks like a MIXED batch (multi-prompt). "
              f"Re-run vLLM bench with NUM_PROMPTS=1 for a clean single-prompt dump.")

    model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device).eval()
    backbone = find_backbone(model)

    ids = vtok.unsqueeze(0).to(args.device)                     # (1, T)
    with torch.no_grad():
        bb = backbone(ids, use_cache=True, return_shared_kv_states=True,
                      output_hidden_states=False)
    thid = bb.last_hidden_state[0].float().cpu()                # (T, 2816)

    # Per-position comparison, aligned token-by-token (same token_ids by construction).
    cos = F.cosine_similarity(thid, vhid, dim=-1)               # (T,)
    tn = thid.norm(dim=-1)
    vn = vhid.norm(dim=-1)
    ratio = tn / vn.clamp_min(1e-6)
    l2diff = (thid - vhid).norm(dim=-1) / tn.clamp_min(1e-6)    # relative L2 error

    print("\n============= STRICT per-position hidden diff =============")
    print(f"cosine sim:   mean={cos.mean():.4f} min={cos.min():.4f} "
          f"p10={cos.kthvalue(max(1,T//10)).values:.4f}")
    print(f"norm ratio train/vllm: mean={ratio.mean():.4f} "
          f"min={ratio.min():.4f} max={ratio.max():.4f}")
    print(f"relative L2 err |train-vllm|/|train|: mean={l2diff.mean():.4f} "
          f"max={l2diff.max():.4f}")
    print(f"train norm mean={tn.mean():.3f}  vllm norm mean={vn.mean():.3f}")
    # show first few tokens explicitly
    print("first 5 positions (cos, train_norm, vllm_norm):")
    for i in range(min(5, T)):
        print(f"  pos {i}: cos={cos[i]:.4f} train={tn[i]:.2f} vllm={vn[i]:.2f} tok={vtok[i].item()}")
    print("----------------------------------------------------------")
    if cos.mean() > 0.99 and abs(ratio.mean() - 1.0) < 0.05:
        print("VERDICT: hidden MATCHES (cos~1, ratio~1). vLLM feeds the draft the same")
        print("  hidden the training target produced -> hidden is NOT the pos0 gap.")
        print("  Next: dump + diff shared_kv (the other draft input).")
    else:
        print("VERDICT: hidden DIVERGES. vLLM feeds a different hidden than training.")
        print("  This IS a pos0 train-vs-deploy mismatch -> align training data-gen.")
    print("==========================================================")


if __name__ == "__main__":
    main()
