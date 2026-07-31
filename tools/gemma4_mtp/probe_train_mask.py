#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
"""
Probe whether the HF Gemma4 assistant leaks future KV when we batch multiple
query positions in ONE forward — the correctness crux for MTP TRAINING.

Inference (get_candidates) runs a SINGLE query position (the last token),
attending to the full target KV prefix. Training wants to process ALL T
positions in parallel for efficiency, but position t must only attend to the
target KV prefix [:t+1]; if the assistant's create_attention_masks instead lets
every query see the whole KV, position t sees FUTURE keys -> information leak ->
inflated train accept that collapses at inference.

This probe builds the target KV once, then compares, for each position t:

  A) SINGLE forward: assistant with q_len=1 at position t, shared_kv sliced to
     [:t+1], position_ids=[[t]]        (this is exactly the inference contract)

  B) PARALLEL forward: assistant with q_len=T (all positions at once), full
     shared_kv, position_ids=[0..T-1]  (what training would do)

If B[:, t] == A for every t -> the mask is correctly causal per position, we can
batch safely. If they differ -> the parallel path leaks; training must supply a
custom causal cross-attention mask (we'll build one).

Run:
    python tools/gemma4_mtp/probe_train_mask.py

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
    ap.add_argument("--seq-text", default="The quick brown fox jumps over the lazy dog today")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    torch_dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Gemma4AssistantForCausalLM,
    )

    # --- build target KV + per-position hidden once (bf16 to fit), then free ---
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, trust_remote_code=True
    ).eval().to(device)
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    ids = tok(args.seq_text, return_tensors="pt").input_ids.to(device)
    T = ids.shape[1]
    with torch.no_grad():
        out = target.model(input_ids=ids, use_cache=True, return_shared_kv_states=True)
    full_kv = {k: (v[0].to(torch_dtype), v[1].to(torch_dtype))
               for k, v in out.shared_kv_states.items()}
    last_hidden = out.last_hidden_state.to(torch_dtype)          # (1,T,2816)
    embed = target.get_input_embeddings()
    tok_emb_all = embed(ids).to(torch_dtype)                     # (1,T,2816)
    del target
    torch.cuda.empty_cache()

    asst = Gemma4AssistantForCausalLM.from_pretrained(
        args.assistant, dtype=torch_dtype, trust_remote_code=True
    ).eval().to(device)

    # inputs_embeds for ALL positions: concat[target_embed(tok_t), target_hidden_t]
    inputs_embeds_all = torch.cat([tok_emb_all, last_hidden], dim=-1)  # (1,T,5632)

    def slice_kv(kv, upto):
        return {k: (v[0][:, :, :upto, :], v[1][:, :, :upto, :]) for k, v in kv.items()}

    print(f"T={T} dtype={args.dtype}")

    # --- A) single-position forwards (inference contract) ---
    single_logits = []
    single_hidden = []
    with torch.no_grad():
        for t in range(T):
            emb_t = inputs_embeds_all[:, t : t + 1, :]                 # (1,1,5632)
            kv_t = slice_kv(full_kv, t + 1)                            # prefix [:t+1]
            pos_t = torch.tensor([[t]], dtype=torch.long, device=device)
            o = asst(inputs_embeds=emb_t, position_ids=pos_t,
                     shared_kv_states=kv_t, use_cache=False)
            single_logits.append(o.logits)                            # (1,1,V)
            single_hidden.append(o.last_hidden_state)                 # (1,1,2816)
    single_logits = torch.cat(single_logits, dim=1)                   # (1,T,V)
    single_hidden = torch.cat(single_hidden, dim=1)                   # (1,T,2816)

    # --- B) parallel forward (naive training path), full KV, all positions ---
    pos_all = torch.arange(T, device=device).unsqueeze(0)
    with torch.no_grad():
        ob = asst(inputs_embeds=inputs_embeds_all, position_ids=pos_all,
                  shared_kv_states=full_kv, use_cache=False)
    par_logits = ob.logits                                            # (1,T,V)
    par_hidden = ob.last_hidden_state                                 # (1,T,2816)

    # --- compare per position ---
    dlog = (par_logits.float() - single_logits.float()).abs()
    dhid = (par_hidden.float() - single_hidden.float()).abs()
    print("\nper-position max_abs_diff (parallel vs single):")
    print("  pos |    logits   |  last_hidden")
    for t in range(T):
        print(f"  {t:3d} | {dlog[0, t].max().item():.3e} | {dhid[0, t].max().item():.3e}")

    max_log = dlog.max().item()
    max_hid = dhid.max().item()
    print(f"\noverall max_abs_diff  logits={max_log:.3e}  last_hidden={max_hid:.3e}")

    tol = 1e-3 if args.dtype == "float32" else 5e-1
    # Position 0 should ALWAYS match (only sees kv[:1] either way). The telling
    # positions are t>0: if parallel differs there, the mask is NOT causal.
    later = dlog[0, 1:].amax(dim=-1) if T > 1 else dlog[0, :1].amax(dim=-1)
    if max_log < tol and max_hid < tol:
        print("RESULT: parallel == single per position -> mask IS causal, "
              "batching over T is SAFE. ✅")
    elif dlog[0, 0].max().item() < tol and later.max().item() >= tol:
        print("RESULT: pos 0 matches but t>0 DIVERGE -> parallel path LEAKS "
              "future KV. Training MUST supply a custom causal cross-attn mask. ❌")
    else:
        print("RESULT: unexpected pattern — inspect the per-position table above. ⚠️")

    print("\n# DONE. Paste everything above back.")


if __name__ == "__main__":
    main()
