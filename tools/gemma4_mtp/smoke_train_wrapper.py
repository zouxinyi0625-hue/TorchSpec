#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
"""
Smoke test for the Gemma4 MTP TRAINING wrapper (torchspec/models/gemma4_mtp.py).

verify_parity proved the draft forward is faithful; this checks the NEW,
unverified training-loop wrapper actually runs end-to-end and produces sane
numbers: finite loss, per-step accept in [0,1], correct output shapes, and a
backward pass that yields gradients on the draft params (not the frozen target).

It builds one real training example from the target (last_hidden per position +
shared_kv + input_ids), runs Gemma4MTPModel.forward, and sanity-checks outputs.

Run (GPU + weights):
    python tools/gemma4_mtp/smoke_train_wrapper.py

Paste stdout back.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="/tmp/models/gemma4/text_only")
    ap.add_argument("--assistant", default="/tmp/models/gemma4/assistant")
    ap.add_argument("--seq-text", default="The quick brown fox jumps over the lazy dog today and tomorrow")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    dt = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- build one training example from the target (bf16 to fit), then free ---
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, trust_remote_code=True
    ).eval().to(device)
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    ids = tok(args.seq_text, return_tensors="pt").input_ids.to(device)
    T = ids.shape[1]
    with torch.no_grad():
        out = target.model(input_ids=ids, use_cache=True, return_shared_kv_states=True)
    target_last_hidden = out.last_hidden_state.to(dt)                 # (1,T,2816)
    shared_kv = {k: (v[0].to(dt), v[1].to(dt)) for k, v in out.shared_kv_states.items()}
    embed_w = target.get_input_embeddings().weight.detach().to(dt).clone()   # (V,2816) scaled
    # tied lm_head == embedding for Gemma4 (tie_word_embeddings=True)
    lm_head_w = target.get_output_embeddings().weight.detach().to(dt).clone()  # (V,2816)
    loss_mask = torch.ones_like(ids, dtype=torch.float32)
    del target
    torch.cuda.empty_cache()

    # --- build draft + training wrapper ---
    from torchspec.models.draft.gemma4_mtp import Gemma4MTPConfig, Gemma4MTPDraftModel
    from torchspec.models.gemma4_mtp import Gemma4MTPModel

    cfg = Gemma4MTPConfig(assistant_model_path=args.assistant, target_model_path=args.target,
                          mtp_num_steps=args.k)
    draft = Gemma4MTPDraftModel(cfg).to(device=device, dtype=dt)
    draft.load_assistant_weights(args.assistant)
    draft.to(device=device, dtype=dt)
    wrapper = Gemma4MTPModel(draft, mtp_num_steps=args.k, teacher_force=True).to(device)

    print(f"T={T} K={args.k} dtype={args.dtype} layer_types={list(shared_kv.keys())}")

    loss, acc, loss_ps, acc_ps, cnt_ps = wrapper(
        input_ids=ids,
        target_last_hidden=target_last_hidden,
        shared_kv_states=shared_kv,
        loss_mask=loss_mask,
        target_embed_weight=embed_w,
        target_lm_head_weight=lm_head_w,
    )

    print(f"\nloss           = {loss.item():.4f}  (finite={torch.isfinite(loss).item()})")
    print(f"accuracy       = {acc.item():.4f}")
    print(f"loss_per_step  = {[round(x, 4) for x in loss_ps.tolist()]}")
    print(f"acc_per_step   = {[round(x, 4) for x in acc_ps.tolist()]}")
    print(f"count_per_step = {[int(x) for x in cnt_ps.tolist()]}")

    # --- backward: grads on draft (trainable), none required on target tensors ---
    loss.backward()
    n_grad = sum(1 for p in draft.parameters() if p.requires_grad and p.grad is not None)
    n_train = sum(1 for p in draft.parameters() if p.requires_grad)
    gnorm = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in draft.parameters()
                           if p.grad is not None)).item()
    print(f"\nbackward OK: {n_grad}/{n_train} trainable params got grad, grad_norm={gnorm:.3e}")

    # --- sanity assertions ---
    ok = True
    if not torch.isfinite(loss).item():
        print("FAIL: loss not finite ❌"); ok = False
    if not (0.0 <= acc.item() <= 1.0):
        print("FAIL: accuracy out of [0,1] ❌"); ok = False
    if not all(0.0 <= a <= 1.0 for a in acc_ps.tolist()):
        print("FAIL: per-step accept out of [0,1] ❌"); ok = False
    if loss_ps.shape[0] != args.k or acc_ps.shape[0] != args.k:
        print(f"FAIL: per-step shape != K ❌"); ok = False
    if n_grad == 0:
        print("FAIL: no gradients on draft ❌"); ok = False
    if gnorm == 0.0:
        print("FAIL: zero grad norm ❌"); ok = False

    # step-0 accept should generally beat later steps (nearer = easier)
    if args.k > 1 and acc_ps[0].item() < acc_ps[-1].item():
        print("NOTE: step-0 accept < last-step accept — worth watching (not fatal).")

    print("\nSMOKE OK ✅" if ok else "\nSMOKE FAILED ❌")
    if not ok:
        raise SystemExit(1)
    print("# DONE. Paste everything above back.")


if __name__ == "__main__":
    main()
