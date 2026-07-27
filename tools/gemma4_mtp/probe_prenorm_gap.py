#!/usr/bin/env python
"""
PROBE: prove the train/deploy hidden-state mismatch that halves vLLM accept.

HYPOTHESIS (from reading the code):
  - TRAINING feeds the draft the TARGET's POST-norm last_hidden_state.
    torchspec/models/target/gemma4_mtp_target.py:175  `last_hidden = out.last_hidden_state`
    -> for HF Gemma4, last_hidden_state is AFTER the backbone final RMSNorm (model.norm).
    And inference_config.py:resolve_last_hidden_states_prenorm() returns
    (inference_engine_type == "vllm"), which is False for engine_type="hf"
    -> prenorm=False -> POST-norm.
  - vLLM DEPLOYMENT feeds the draft the TARGET's PRE-norm hidden (raw last layer
    output). Its own gemma4_mtp.py docstring / the connector only captures raw
    layer outputs (pre-norm).

  => draft trained on post-norm hidden (RMSNorm'd, ~O(1) magnitude) but served
     pre-norm hidden (un-normalized, larger magnitude) -> pre_projection sees an
     out-of-distribution input -> every position's draft prediction is off ->
     accept halves UNIFORMLY (pos0 included). Matches the observed 0.86->0.43.

WHAT THIS PROBE DOES (falsifiable, ~30s on the server, 1 GPU):
  Load the 26B target, run ONE real layer1_delta prompt, and print the per-token
  norm of:
    (a) PRE-norm  hidden  = raw output of the last decoder layer (what vLLM feeds)
    (b) POST-norm hidden  = model.norm(pre_norm) = out.last_hidden_state (what training fed)
  If ||pre|| and ||post|| differ substantially (expected: pre >> post, since
  RMSNorm pulls magnitude to ~sqrt(hidden)), the mismatch is REAL and is the
  deployment gap. If they're ~equal, the hypothesis is WRONG -- look elsewhere.

RUN ON THE SERVER (1 GPU is enough), from the TorchSpec root:
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/probe_prenorm_gap.py \
      --target /tmp/models/gemma4/text_only \
      --data   data/eval_layer1_delta_1000.jsonl \
      --max-len 2048
"""
from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="Path to the 26B text_only target model dir")
    ap.add_argument("--data", required=True, help="layer1_delta eval jsonl (conversations schema)")
    ap.add_argument("--max-len", type=int, default=2048, help="Truncate prompt to this many tokens")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer + target from {args.target} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device).eval()

    # Locate the backbone + its final norm. HF Gemma4 text model exposes
    # model.model (the decoder stack) with .norm as the final RMSNorm.
    backbone = model.model if hasattr(model, "model") else model
    final_norm = backbone.norm
    print(f"final norm module: {type(final_norm).__name__}, "
          f"weight shape {tuple(final_norm.weight.shape)}", flush=True)

    # Build one real prompt (system+user folded, drop assistant) from row 0.
    with open(args.data, encoding="utf-8") as f:
        rec = json.loads(f.readline())
    convs = rec["conversations"]
    sys_c = next((t["content"] for t in convs if t["role"] == "system"), "")
    usr_c = next((t["content"] for t in convs if t["role"] == "user"), "")
    text = (sys_c + "\n\n" + usr_c) if sys_c else usr_c
    ids = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len).input_ids.to(args.device)
    print(f"prompt tokens: {ids.shape[1]}", flush=True)

    # Forward with output_hidden_states so we get the PRE-norm last layer output.
    # hidden_states[-1] is the raw output of the last decoder layer (PRE final-norm).
    # out.last_hidden_state is that same tensor AFTER model.norm (POST-norm).
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)

    pre_norm = out.hidden_states[-1]                 # what vLLM feeds the draft
    # Reproduce the post-norm the training target used: out.last_hidden_state.
    # (Some HF versions already norm hidden_states[-1]; recompute explicitly to be safe.)
    post_norm_recomputed = final_norm(pre_norm)      # what training fed the draft

    # Per-token L2 norm across hidden dim, then average over tokens.
    def mean_tok_norm(x: torch.Tensor) -> float:
        return x[0].float().norm(dim=-1).mean().item()

    pre_n = mean_tok_norm(pre_norm)
    post_n = mean_tok_norm(post_norm_recomputed)

    print("\n================= PRENORM-GAP PROBE RESULT =================")
    print(f"PRE-norm  hidden mean per-token L2 norm (vLLM feeds this):  {pre_n:8.3f}")
    print(f"POST-norm hidden mean per-token L2 norm (training fed this): {post_n:8.3f}")
    print(f"ratio pre/post: {pre_n / max(post_n, 1e-9):8.3f}")
    print("-----------------------------------------------------------")
    if pre_n / max(post_n, 1e-9) > 1.5 or post_n / max(pre_n, 1e-9) > 1.5:
        print("VERDICT: SUBSTANTIAL mismatch -> train(post-norm) vs deploy(pre-norm)")
        print("         gap CONFIRMED. This is why vLLM accept halves.")
        print("         Fix A: retrain with inference.last_hidden_states_prenorm=true")
        print("         Fix B: apply model.norm to hidden_states in vLLM gemma4_mtp before pre_projection")
    else:
        print("VERDICT: norms are CLOSE -> prenorm/postnorm is NOT the gap. Look elsewhere.")
    print("===========================================================")


if __name__ == "__main__":
    main()
