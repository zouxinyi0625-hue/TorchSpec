#!/usr/bin/env python
"""
Test the pos0 hypothesis: does the trained draft collapse when fed vLLM's
hidden instead of HF's hidden? (i.e. did it overfit the exact HF-precision
hidden and fail to generalize to vLLM's 16%-offset hidden?)

Setup: hold EVERYTHING constant except the prev_hidden fed to the draft.
  - shared_kv: HF-computed (same for both arms)
  - token: from the vLLM dump (same for both arms)
  - hidden arm A: HF target last_hidden (what training/eval used)
  - hidden arm B: vLLM's dumped target_hidden_states (what deploy feeds)
Then measure the draft's pos0 top-1 agreement with target-from-hidden argmax
(same acc definition as training) for each arm.

If arm A ~0.9 and arm B ~0.4, the draft is hidden-precision-sensitive: it
overfit HF hidden and collapses on vLLM hidden -> THE pos0 root cause. The fix
is on the training side (train on hidden that matches vLLM's, or make the draft
robust to the offset).

USAGE (server, 1 GPU), TorchSpec root:
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/test_hidden_sensitivity.py \
      --target    /tmp/models/gemma4/text_only \
      --assistant $AZURE_ML_INPUT_UKWDATA/maiprofile/mtp_26b/gemma4_mtp_exp1_hf/hf_iter_0002269 \
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
    ap.add_argument("--assistant", required=True)
    ap.add_argument("--vllm-dump", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import (
        AutoModelForCausalLM,
        Gemma4AssistantForCausalLM,
    )

    dev = args.device
    v = torch.load(args.vllm_dump, map_location="cpu")
    vtok = v["target_token_ids"].reshape(-1).to(torch.long)
    vhid = v["target_hidden_states"].float().to(dev)             # (T, 2816) vLLM hidden
    T = vtok.shape[0]
    ids = vtok.unsqueeze(0).to(dev)
    print(f"vLLM dump: {T} tokens")

    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(dev).eval()
    backbone = find_backbone(target)
    target_embed = backbone.embed_tokens.weight                 # (V, 2816)
    embed_scale = torch.tensor(target_embed.shape[1] ** 0.5, dtype=target_embed.dtype, device=dev)

    with torch.no_grad():
        bb = backbone(ids, use_cache=True, return_shared_kv_states=True, output_hidden_states=False)
    hf_hid = bb.last_hidden_state[0].float().to(dev)             # (T, 2816) HF hidden
    shared_kv = bb.shared_kv_states                              # HF kv, used for BOTH arms

    draft = Gemma4AssistantForCausalLM.from_pretrained(
        args.assistant, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(dev).eval()

    # target_p argmax (the acc reference), from HF hidden -- the "truth" both arms
    # try to predict (same as training acc: argmax(target_lm_head(target_hidden))).
    with torch.no_grad():
        target_pred = (hf_hid @ target_embed.float().T).argmax(-1)   # (T,)

    def run_arm(prev_hidden, name):
        cur_token = ids.clamp(0, target_embed.shape[0] - 1)
        tok_embed = F.embedding(cur_token, target_embed) * embed_scale        # (1,T,2816)
        inputs_embeds = torch.cat([tok_embed, prev_hidden.unsqueeze(0).to(tok_embed.dtype)], dim=-1)
        position_ids = torch.arange(T, device=dev).unsqueeze(0)
        with torch.no_grad():
            out = draft(inputs_embeds=inputs_embeds, position_ids=position_ids,
                        shared_kv_states=shared_kv)
        dg = out.logits[0].argmax(-1)                            # (T,)
        agree = (dg == target_pred).float().mean().item()
        hn = prev_hidden.norm(dim=-1).mean().item()
        print(f"  arm {name}: pos-agreement={agree:.4f}  (prev_hidden L2 mean={hn:.2f})")
        return agree

    print("\n============ HIDDEN-SENSITIVITY TEST (trained draft) ============")
    a = run_arm(hf_hid, "A: HF hidden   (train/eval used)")
    b = run_arm(vhid,   "B: vLLM hidden (deploy feeds) ")
    print("----------------------------------------------------------------")
    print(f"delta (A - B) = {a - b:.4f}")
    if a - b > 0.15:
        print("VERDICT: draft COLLAPSES on vLLM hidden -> hidden-precision-sensitive.")
        print("  The pos0 gap is the draft overfitting HF-precision hidden. Fix on the")
        print("  TRAINING side: generate training hidden that matches vLLM's, or add")
        print("  robustness (noise on hidden) so it generalizes across inference stacks.")
    elif abs(a - b) <= 0.15 and a < 0.6:
        print("VERDICT: draft is LOW on BOTH -> not hidden-precision; the acc definition")
        print("  or shared_kv differs. Investigate shared_kv / acc harness next.")
    else:
        print("VERDICT: draft robust to hidden source (A~=B, both high). pos0 gap is")
        print("  elsewhere (shared_kv or the vLLM multi-step path). Investigate shared_kv.")
    print("================================================================")


if __name__ == "__main__":
    main()
