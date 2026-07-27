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

    # Locate the backbone + its final norm. HF Gemma4 nests the decoder stack;
    # the final RMSNorm attribute isn't always `.norm` on the top module, so
    # search for it robustly.
    def find_backbone_and_norm(m):
        import torch.nn as nn
        # Walk down common nesting: model -> (language_model|model) -> ... until
        # we find a module that has a `.norm` RMSNorm and a `.layers`.
        candidates = [m]
        seen = set()
        while candidates:
            mod = candidates.pop(0)
            if id(mod) in seen:
                continue
            seen.add(id(mod))
            has_norm = hasattr(mod, "norm") and isinstance(getattr(mod, "norm"), nn.Module)
            has_layers = hasattr(mod, "layers")
            if has_norm and has_layers:
                return mod, getattr(mod, "norm")
            for _n, child in mod.named_children():
                candidates.append(child)
        return None, None

    backbone, final_norm = find_backbone_and_norm(model)
    if final_norm is None:
        # Dump structure to help pin the attribute name, then bail.
        print("Could NOT auto-locate final norm. Top-level module tree:", flush=True)
        for name, mod in model.named_modules():
            if name.count(".") <= 2 and ("norm" in name.lower() or name.endswith("layers")):
                print(f"  {name}: {type(mod).__name__}", flush=True)
        raise SystemExit("Set the norm path manually from the tree above.")
    print(f"backbone: {type(backbone).__name__}, "
          f"final norm: {type(final_norm).__name__}, "
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

    last_layer_hs = out.hidden_states[-1]            # last entry of hidden_states tuple
    lhs = out.last_hidden_state                      # what training used (gemma4_mtp_target.py:175)
    post_of_lastlayer = final_norm(last_layer_hs)    # explicitly norm the last-layer output

    # Per-token L2 norm across hidden dim, averaged over tokens.
    def mean_tok_norm(x: torch.Tensor) -> float:
        return x[0].float().norm(dim=-1).mean().item()

    n_lastlayer = mean_tok_norm(last_layer_hs)
    n_lhs = mean_tok_norm(lhs)
    n_post = mean_tok_norm(post_of_lastlayer)
    # Is out.last_hidden_state already normed (== post_of_lastlayer)?
    lhs_is_normed = abs(n_lhs - n_post) / max(n_post, 1e-9) < 0.05

    print("\n================= PRENORM-GAP PROBE RESULT =================")
    print(f"hidden_states[-1]  (raw last-layer)          mean L2: {n_lastlayer:9.3f}")
    print(f"out.last_hidden_state (TRAINING fed this)    mean L2: {n_lhs:9.3f}")
    print(f"norm(hidden_states[-1]) (explicit post-norm) mean L2: {n_post:9.3f}")
    print("-----------------------------------------------------------")
    print(f"is out.last_hidden_state already norm'd? {lhs_is_normed} "
          f"(|lhs-post|/post = {abs(n_lhs-n_post)/max(n_post,1e-9):.3f})")
    print(f"ratio raw_lastlayer / last_hidden_state: {n_lastlayer/max(n_lhs,1e-9):9.3f}")
    print("-----------------------------------------------------------")
    # The deployment mismatch = (what vLLM feeds: raw pre-norm last-layer)
    #                       vs (what training fed: out.last_hidden_state).
    ratio = n_lastlayer / max(n_lhs, 1e-9)
    if ratio > 1.5 or ratio < 0.667:
        print("VERDICT: SUBSTANTIAL mismatch between raw last-layer (vLLM pre-norm)")
        print("         and out.last_hidden_state (training). GAP CONFIRMED.")
        print("         -> vLLM feeds un-normalized hidden; draft trained on normalized.")
        print("         Fix A: retrain with inference.last_hidden_states_prenorm=true")
        print("         Fix B: apply model.norm in vLLM gemma4_mtp before pre_projection")
    else:
        print("VERDICT: raw last-layer ~= last_hidden_state -> norm is NOT the gap.")
        print("         (out.last_hidden_state may itself be pre-norm here.) Look elsewhere:")
        print("         next probe = per-layer draft-forward parity HF-assistant vs vLLM.")
    print("===========================================================")


if __name__ == "__main__":
    main()
