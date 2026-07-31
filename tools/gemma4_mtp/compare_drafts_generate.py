#!/usr/bin/env python
"""
FAITHFUL HF acceptance comparison via the OFFICIAL assisted-generation API.

Instead of hand-rolling the draft forward (which got pre/post-norm and the
autoregressive single-token contract wrong), this drives transformers' real
`target.generate(assistant_model=draft)` -- which internally uses
SinglePositionMultiTokenCandidateGenerator (the correct Gemma4 MTP path:
hidden_states[-1] pre-norm, single-token autoregressive, draft feeds its own
hidden back). We monkey-patch `_speculative_sampling` to tally the true
per-step accepted count -> real acceptance rate & length, official draft vs
your trained draft, same target, same prompts.

USAGE (server):
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/compare_drafts_generate.py \
      --target   $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
      --official $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
      --trained  $AZURE_ML_INPUT_UKWDATA/maiprofile/mtp_26b/gemma4_mtp_exp1_hf/hf_iter_0002269 \
      --data     data/eval_layer1_delta_1000.jsonl --num-prompts 5 \
      --gen-len 128 --num-assistant-tokens 5
"""
from __future__ import annotations

import argparse
import json

import torch

# Global tally updated by the patched candidate generator.
STATS = {"accepted": 0, "proposed": 0, "steps": 0, "rounds": 0, "last_proposed": 0}


def install_patch():
    """Tally accepts/proposed the SAME way vLLM does:
       proposed = draft tokens offered per round (candidate len)
       accepted = matched tokens that round (num_matches)
    Hook update_candidate_strategy(input_ids, scores, num_matches) for accepts,
    and get_candidates return for proposed length."""
    from transformers.generation import candidate_generator as CG

    spm = getattr(CG, "SinglePositionMultiTokenCandidateGenerator", None)
    if spm is None:
        return

    orig_gc = spm.get_candidates

    def patched_gc(self, input_ids, *a, **k):
        cand_ids, cand_logits = orig_gc(self, input_ids, *a, **k)
        proposed = int(cand_ids.shape[1] - input_ids.shape[1])
        STATS["proposed"] += max(proposed, 0)
        STATS["rounds"] += 1
        STATS["last_proposed"] = max(proposed, 0)
        return cand_ids, cand_logits

    spm.get_candidates = patched_gc

    if "update_candidate_strategy" in spm.__dict__:
        orig_us = spm.update_candidate_strategy

        def patched_us(self, input_ids, scores, num_matches):
            STATS["accepted"] += int(num_matches)
            STATS["steps"] += 1
            return orig_us(self, input_ids, scores, num_matches)

        spm.update_candidate_strategy = patched_us


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--num-assistant-tokens", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    install_patch()

    from transformers import AutoModelForCausalLM, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.target)
    tok = getattr(proc, "tokenizer", proc)

    dev = args.device
    print(f"loading target on {dev} ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype="auto"
    ).to(dev).eval()

    print("loading drafts (same device) ...")
    drafts = {
        "official": AutoModelForCausalLM.from_pretrained(
            args.official, dtype="auto").to(dev).eval(),
        "trained": AutoModelForCausalLM.from_pretrained(
            args.trained, dtype="auto").to(dev).eval(),
    }

    lines = [l for l in open(args.data, encoding="utf-8") if l.strip()][: args.num_prompts]

    results = {}
    for name, draft in drafts.items():
        # Do NOT override num_assistant_tokens -- use the draft's own generation
        # config, exactly like the official usage.

        STATS["accepted"] = STATS["proposed"] = STATS["steps"] = STATS["rounds"] = 0
        gen_tokens = 0
        for i, line in enumerate(lines):
            rec = json.loads(line)
            convs = rec["conversations"]
            prompt_msgs = [{"role": ("system" if m.get("role") == "system" else "user"),
                            "content": m["content"]}
                           for m in convs if m.get("role") in ("system", "user")]
            inputs = proc.apply_chat_template(
                prompt_msgs, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True, enable_thinking=False,
            ).to(target.device)
            input_len = inputs["input_ids"].shape[-1]
            if input_len > args.max_len:
                continue
            with torch.no_grad():
                out = target.generate(
                    **inputs,
                    assistant_model=draft,
                    max_new_tokens=args.gen_len, do_sample=False,
                )
            gen_tokens += int(out.shape[1] - input_len)
        # Derive accepts from generation, no internal hooks needed:
        # each drafting round produces (accepted + 1) tokens, so
        #   total_accepted = gen_tokens - rounds
        #   accept_rate    = total_accepted / proposed
        #   accept_len     = gen_tokens / rounds   (accepted + 1 per round)
        prop = STATS["proposed"]
        rounds = STATS["rounds"]
        accepted = max(gen_tokens - rounds, 0)
        rate = accepted / max(prop, 1)
        length = gen_tokens / max(rounds, 1)
        avg_proposed = prop / max(rounds, 1)
        results[name] = (rate, length, accepted, prop, rounds, gen_tokens, avg_proposed)
        print(f"  {name:9}: accept_rate={rate:.4f}  accept_len={length:.2f}  "
              f"avg_proposed/round={avg_proposed:.2f}  "
              f"(accepted={accepted} proposed={prop} rounds={rounds} gen={gen_tokens})")

    print("=" * 60)
    print("REAL assisted-generation acceptance (official HF API, same target):")
    for name in ("official", "trained"):
        r, l, a, p, s, g, ap = results[name]
        print(f"  {name:9}: accept_rate={r:.4f}  accept_len={l:.2f}  avg_proposed={ap:.2f}")
    ro = results["official"][0]
    rt = results["trained"][0]
    print("-" * 60)
    print(f"delta (official - trained) = {ro - rt:.4f}")
    print("official should reproduce its known-good level here; if trained is far")
    print("below official in HF too -> training; if trained ~ official in HF but")
    print("far below in vLLM deploy -> vLLM engine path.")
    print("=" * 60)


if __name__ == "__main__":
    main()
