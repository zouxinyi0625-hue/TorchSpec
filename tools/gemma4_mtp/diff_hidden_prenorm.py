#!/usr/bin/env python
"""
Verify the strongest pos0 suspect: does the target's last_hidden_state (POST-norm,
what TRAINING feeds the draft) differ from hidden_states[-1] (what the official
deploy candidate generator feeds -- SinglePositionMultiTokenCandidateGenerator
sets output_hidden_states=True and uses model_outputs.hidden_states[-1])?

If hidden_states[-1] is PRE-norm while training uses POST-norm, that's a real
train-vs-deploy mismatch and explains pos0 collapsing the most (pos0 is the only
step fed the raw target hidden with no draft-side accumulation).

USAGE:
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/diff_hidden_prenorm.py \
      --target $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
      --data data/eval_layer1_delta_1000.jsonl
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, trust_remote_code=True
    ).to(dev).eval()

    line = next(l for l in open(args.data, encoding="utf-8") if l.strip())
    rec = json.loads(line)
    msgs = [{"role": ("system" if m.get("role") == "system" else
                      ("assistant" if m.get("role") in ("assistant", "model") else "user")),
             "content": m["content"]} for m in rec["conversations"]]
    ids = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt")
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    ids = ids[:, : args.max_len].to(dev)

    with torch.no_grad():
        out = model.model(input_ids=ids, use_cache=True,
                          return_shared_kv_states=True,
                          output_hidden_states=True)

    post_norm = out.last_hidden_state                      # training uses this
    hs = getattr(out, "hidden_states", None)
    print(f"last_hidden_state (post-norm) shape={tuple(post_norm.shape)}")
    if hs is None:
        print("output_hidden_states returned None -> base model doesn't collect them; "
              "deploy candidate generator's hidden_states[-1] would come from the "
              "ForCausalLM wrapper. Re-check at that level.")
        return
    print(f"hidden_states tuple len={len(hs)} (0=embed .. -1=last layer)")
    last_layer = hs[-1]                                    # deploy candidate uses [-1]

    a = post_norm.float().reshape(-1, post_norm.shape[-1])
    b = last_layer.float().reshape(-1, last_layer.shape[-1])
    cos = F.cosine_similarity(a, b, dim=-1)
    rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-6)
    print("=" * 60)
    print("post-norm (train)  vs  hidden_states[-1] (deploy candidate)")
    print(f"  cosine   mean={cos.mean():.4f} min={cos.min():.4f}")
    print(f"  rel L2   mean={rel.mean():.4f} max={rel.max():.4f}")
    print(f"  norm post-norm mean={a.norm(dim=-1).mean():.2f}  "
          f"hs[-1] mean={b.norm(dim=-1).mean():.2f}")
    print("-" * 60)
    if cos.mean() > 0.999 and abs(1 - (b.norm(dim=-1).mean() / a.norm(dim=-1).mean())) < 0.02:
        print("SAME -> hidden_states[-1] is also post-norm; norm is NOT the pos0 cause.")
    else:
        print("DIFFER -> deploy feeds a different (pre-norm?) hidden than training's")
        print("post-norm. THIS is the pos0 mismatch. Fix training to use hidden_states[-1].")
    print("=" * 60)


if __name__ == "__main__":
    main()
