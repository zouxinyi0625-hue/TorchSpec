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

# Global tally updated by the patched _speculative_sampling.
STATS = {"accepted": 0, "proposed": 0, "steps": 0}


def install_patch():
    import transformers.generation.utils as U

    orig = U._speculative_sampling

    def patched(candidate_input_ids, candidate_logits, candidate_length,
                new_logits, is_done_candidate):
        valid_tokens, n_matches = orig(candidate_input_ids, candidate_logits,
                                       candidate_length, new_logits, is_done_candidate)
        STATS["accepted"] += int(n_matches)
        STATS["proposed"] += int(candidate_length)
        STATS["steps"] += 1
        return valid_tokens, n_matches

    U._speculative_sampling = patched


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
    args = ap.parse_args()

    install_patch()

    from transformers import AutoModelForCausalLM, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.target)
    tok = getattr(proc, "tokenizer", proc)

    print("loading target ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype="auto", device_map="auto"
    ).eval()

    print("loading drafts ...")
    drafts = {
        "official": AutoModelForCausalLM.from_pretrained(
            args.official, dtype="auto", device_map="auto").eval(),
        "trained": AutoModelForCausalLM.from_pretrained(
            args.trained, dtype="auto", device_map="auto").eval(),
    }

    lines = [l for l in open(args.data, encoding="utf-8") if l.strip()][: args.num_prompts]

    results = {}
    for name, draft in drafts.items():
        try:
            draft.generation_config.num_assistant_tokens = args.num_assistant_tokens
            draft.generation_config.num_assistant_tokens_schedule = "constant"
        except Exception:
            pass

        STATS["accepted"] = STATS["proposed"] = STATS["steps"] = 0
        for i, line in enumerate(lines):
            rec = json.loads(line)
            convs = rec["conversations"]
            prompt_msgs = [{"role": ("user" if m.get("role") not in ("system",) else "system"),
                            "content": m["content"]}
                           for m in convs if m.get("role") in ("system", "user")]
            inputs = proc.apply_chat_template(
                prompt_msgs, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True,
            )
            ids = inputs["input_ids"][:, : args.max_len].to(target.device)
            am = inputs.get("attention_mask")
            am = am[:, : args.max_len].to(target.device) if am is not None else None
            with torch.no_grad():
                target.generate(
                    input_ids=ids, attention_mask=am,
                    assistant_model=draft,
                    max_new_tokens=args.gen_len, do_sample=False,
                )
        acc, prop, steps = STATS["accepted"], STATS["proposed"], STATS["steps"]
        rate = acc / max(prop, 1)
        length = acc / max(steps, 1)
        results[name] = (rate, length, acc, prop, steps)
        print(f"  {name:9}: accept_rate={rate:.4f}  accept_len={length:.2f}  "
              f"({acc}/{prop} over {steps} steps)")

    print("=" * 60)
    print("REAL assisted-generation acceptance (official HF API, same target):")
    for name in ("official", "trained"):
        r, l, a, p, s = results[name]
        print(f"  {name:9}: accept_rate={r:.4f}  accept_len={l:.2f}")
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
