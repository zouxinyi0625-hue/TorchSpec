#!/usr/bin/env python
"""
Dump the TRAINING-side target output for pos0 and DIFF against vLLM's dump.

Pairs with the read-only dump added to vLLM's llm_base_proposer.propose
(enabled via VLLM_MTP_DUMP_DIR). Run vLLM bench once with that env set to get
vllm_pos0_inputs.pt, then run this to produce the training-side tensors for the
SAME prompt and print a numeric diff of what the draft is fed at step 0:

  - target_hidden_states (the prev_hidden fed to the draft's pre_projection)
  - the last real token id (pos0's input token)

If the hidden norms / values diverge substantially, the deploy pos0 drop is a
real train-vs-deploy input mismatch -> fix the TRAINING data-gen to match vLLM.

USAGE (server, 1 GPU), from TorchSpec root:
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/dump_and_diff_pos0.py \
      --target /tmp/models/gemma4/text_only \
      --data   data/eval_layer1_delta_1000.jsonl \
      --vllm-dump /tmp/mtp_dump/vllm_pos0_inputs.pt \
      --max-len 2048

NOTE: the vLLM dump aggregates the whole scheduled batch; this compares the
LAST-token hidden (pos0 anchor) statistics. First confirm the norm scale and
that the same token id lines up; that alone reveals a gross mismatch.
"""
from __future__ import annotations

import argparse
import json

import torch


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
    ap.add_argument("--data", required=True)
    ap.add_argument("--vllm-dump", default=None, help="path to vllm_pos0_inputs.pt")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device).eval()
    backbone = find_backbone(model)

    # Same prompt construction the bench used: system+user folded, drop assistant.
    with open(args.data, encoding="utf-8") as f:
        rec = json.loads(f.readline())
    convs = rec["conversations"]
    sys_c = next((t["content"] for t in convs if t["role"] == "system"), "")
    usr_c = next((t["content"] for t in convs if t["role"] == "user"), "")
    text = (sys_c + "\n\n" + usr_c) if sys_c else usr_c
    ids = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len).input_ids.to(args.device)
    T = ids.shape[1]

    with torch.no_grad():
        bb = backbone(ids, use_cache=True, return_shared_kv_states=True, output_hidden_states=False)
    train_hidden = bb.last_hidden_state[0].float().cpu()   # (T, 2816) post-norm
    train_hn = train_hidden.norm(dim=-1)

    print("================= TRAINING-side pos0 dump =================")
    print(f"prompt tokens: {T}")
    print(f"train target_hidden (post-norm) per-tok L2: mean={train_hn.mean():.3f} "
          f"last={train_hn[-1]:.3f}")
    print(f"last token id: {ids[0, -1].item()}")

    if args.vllm_dump:
        v = torch.load(args.vllm_dump, map_location="cpu")
        vh = v["target_hidden_states"].float()          # (num_tokens, hidden)
        vhn = vh.norm(dim=-1)
        print("\n================= vLLM-side pos0 dump =================")
        print(f"vllm hidden shape: {tuple(vh.shape)}  hidden_size={v.get('hidden_size')}")
        print(f"vllm target_hidden per-tok L2: mean={vhn.mean():.3f} last={vhn[-1]:.3f}")
        vt = v["target_token_ids"]
        print(f"vllm last token id: {vt.reshape(-1)[-1].item()}")
        print("\n----------------- DIFF -----------------")
        print(f"hidden dim: train=2816 vllm={vh.shape[-1]}  "
              f"{'MATCH' if vh.shape[-1] == 2816 else 'MISMATCH!'}")
        print(f"per-tok L2 mean: train={train_hn.mean():.3f} vllm={vhn.mean():.3f}  "
              f"ratio={train_hn.mean()/max(vhn.mean(),1e-6):.3f}")
        print("If ratio is far from 1.0, the draft is fed a differently-scaled")
        print("hidden at deploy than at train -> that's the pos0 root cause.")
    print("==========================================================")


if __name__ == "__main__":
    main()
