#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
"""
EVIDENCE probe for the MTP label alignment (which shift is correct?).

Claim under test: at anchor position t, fed (token_t, target_hidden_t) with
shared_kv[:t+1], the draft predicts token_{t+1}; so its logits should match the
target's next-token distribution softmax(lm_head(target_hidden_t)) — i.e. the
supervising hidden is target_hidden[t] (shift 0 at step 0), NOT target_hidden[t+1].

We do NOT trust reasoning. We run the draft EXACTLY as inference (single
position, faithful to get_candidates) and measure top-1 agreement with the
target's greedy next token under BOTH hypotheses:

  H0 (shift=0): draft argmax  vs  argmax(lm_head(target_hidden[t]))   -> token_{t+1}
  H1 (shift=1): draft argmax  vs  argmax(lm_head(target_hidden[t+1])) -> token_{t+2}

Whichever hypothesis has HIGH agreement is the correct alignment. Since the
draft weights ARE the pretrained assistant, H_correct should be high (0.6+),
H_wrong near chance. This settles the off-by-one with data, not opinion.

Run (GPU + weights):
    python tools/gemma4_mtp/probe_label_alignment.py

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
    ap.add_argument("--seq-text", default=(
        "The capital of France is Paris and the capital of Japan is Tokyo. "
        "Water boils at one hundred degrees celsius at sea level."))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16

    from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma4AssistantForCausalLM

    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dt, trust_remote_code=True
    ).eval().to(device)
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    ids = tok(args.seq_text, return_tensors="pt").input_ids.to(device)
    T = ids.shape[1]

    with torch.no_grad():
        out = target.model(input_ids=ids, use_cache=True, return_shared_kv_states=True)
    full_kv = out.shared_kv_states
    target_hidden = out.last_hidden_state              # (1,T,2816)
    embed = target.get_input_embeddings()
    lm_head_w = target.get_output_embeddings().weight  # (V,2816) tied

    # Target greedy next-token from each position's hidden (ground-truth alignment).
    with torch.no_grad():
        target_greedy = torch.matmul(target_hidden.float(),
                                     lm_head_w.float().t()).argmax(-1)  # (1,T) token_{t+1} pred by hidden_t

    asst = Gemma4AssistantForCausalLM.from_pretrained(
        args.assistant, dtype=dt, trust_remote_code=True
    ).eval().to(device)

    def slice_kv(kv, upto):
        return {k: (v[0][:, :, :upto, :], v[1][:, :, :upto, :]) for k, v in kv.items()}

    # Run draft faithfully per position (STEP 0 of the MTP round): fed token_t and
    # target_hidden_t, shared_kv[:t+1], position_ids=[[t]]. Collect its argmax.
    draft_pred = []
    with torch.no_grad():
        for t in range(T):
            tok_emb = embed(ids[:, t:t+1])                     # (1,1,2816)
            prev_hidden = target_hidden[:, t:t+1, :]           # (1,1,2816)
            inputs_embeds = torch.cat([tok_emb, prev_hidden], dim=-1)
            o = asst(inputs_embeds=inputs_embeds,
                     position_ids=torch.tensor([[t]], device=device),
                     shared_kv_states=slice_kv(full_kv, t+1), use_cache=False)
            draft_pred.append(o.logits.argmax(-1))            # (1,1)
    draft_pred = torch.cat(draft_pred, dim=1)                 # (1,T)

    # Hypotheses (only compare where both sides in-bounds).
    # H0: draft_pred[t] should equal target_greedy[t]      (= token_{t+1})
    # H1: draft_pred[t] should equal target_greedy[t+1]    (= token_{t+2})
    dp = draft_pred[0]
    tg = target_greedy[0]
    ids0 = ids[0]

    # agreement of draft with target greedy under H0 / H1
    h0 = (dp[:T-1] == tg[:T-1]).float().mean().item()
    h1 = (dp[:T-1] == tg[1:T]).float().mean().item()

    # also: does draft predict the ACTUAL next token in the text? (upper-ish bound)
    acc_actual_next = (dp[:T-1] == ids0[1:T]).float().mean().item()

    print(f"T={T}")
    print(f"H0  draft vs target_greedy[t]   (shift=0, our fix): agree={h0:.3f}")
    print(f"H1  draft vs target_greedy[t+1] (shift=1, old code): agree={h1:.3f}")
    print(f"aux draft vs actual next token in text            : agree={acc_actual_next:.3f}")
    print()
    if h0 > h1 + 0.15:
        print("EVIDENCE: shift=0 (our fix) is correct. Draft@t predicts token_{t+1}. ✅")
    elif h1 > h0 + 0.15:
        print("EVIDENCE: shift=1 (old code) was correct — our fix is WRONG, revert. ❌")
    else:
        print("EVIDENCE: inconclusive/close — inspect a few positions below. ⚠️")

    # Show first several positions for manual inspection.
    print("\n pos | draft_pred | tgt_greedy[t] | tgt_greedy[t+1] | actual_next")
    for t in range(min(T - 1, 12)):
        print(f" {t:3d} | {dp[t].item():10d} | {tg[t].item():13d} | "
              f"{tg[t+1].item():15d} | {ids0[t+1].item():10d}")

    print("\n# DONE. Paste everything above back.")


if __name__ == "__main__":
    main()
