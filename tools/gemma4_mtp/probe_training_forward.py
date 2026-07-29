#!/usr/bin/env python
"""
Probe the TRAINING forward (Gemma4MTPModel) with OFFICIAL assistant weights, to
confirm the training pipeline (strip forward + token/label off-by-one shift) is
aligned with vLLM BEFORE launching a real training run.

Logic: the official draft is known-good (vLLM pos0 ~0.84). If our training
forward is correctly aligned, running the official weights through it on real
target data should give a HIGH step-0 acc (draft argmax == target argmax). A low
pos0 here would mean the training forward/label alignment is still wrong — catch
it now, not after a training run.

Data source: HF Gemma4 target (return_shared_kv_states=True) on one real prompt.
We reuse the strip dump's target_token_ids as the input sequence so it's a real
layer1 example, but any prompt works.

USAGE (server, GPU, transformers 5.9):
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/probe_training_forward.py \
      --official $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
      --target   $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
      --dump /tmp/vllm_draft_step0.pt
"""
from __future__ import annotations

import argparse

import torch


def _find_embed(m):
    q = [m]
    while q:
        x = q.pop(0)
        if hasattr(x, "embed_tokens"):
            return x.embed_tokens.weight
        q.extend(list(x.children()))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--data", help="eval jsonl (conversations); overrides --dump")
    ap.add_argument("--prompt-key", default="conversations")
    ap.add_argument("--num-prompts", type=int, default=3)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--dump", default="/tmp/vllm_draft_step0.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--k", type=int, default=4, help="MTP num steps")
    ap.add_argument("--teacher-force", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torchspec.models.draft.gemma4_mtp import Gemma4MTPConfig, Gemma4MTPDraftModel
    from torchspec.models.gemma4_mtp import Gemma4MTPModel

    dev = args.device

    # ---- build the list of prompt token sequences ----
    seqs = []
    if args.data:
        import json
        tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
        with open(args.data) as f:
            for line in f:
                if len(seqs) >= args.num_prompts:
                    break
                row = json.loads(line)
                convs = row[args.prompt_key]
                ids = tok.apply_chat_template(
                    convs, tokenize=True, add_generation_prompt=False,
                )
                ids = torch.tensor(ids[: args.max_seq], device=dev).long().view(1, -1)
                if ids.shape[1] >= 8:
                    seqs.append(ids)
        print(f"loaded {len(seqs)} prompts from {args.data}")
    else:
        d = torch.load(args.dump, map_location="cpu")
        tgt_ids = d.get("target_token_ids")
        if tgt_ids is None:
            tgt_ids = d["input_ids"]
        seqs.append(tgt_ids.to(dev).long().view(1, -1))
        print(f"loaded 1 prompt from dump T={seqs[0].shape[1]}")

    # ---- HF target: hidden + shared_kv on the real sequence ----
    print("loading HF target ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    embed_w = _find_embed(target).to(dev)
    lm_head_w = embed_w  # Gemma tied embeddings

    inner = target
    for attr in ("model", "language_model"):
        nxt = getattr(inner, attr, None)
        if nxt is not None and hasattr(nxt, "forward"):
            inner = nxt

    # ---- build the training model with OFFICIAL assistant weights ----
    print("loading OFFICIAL draft ...")
    cfg = Gemma4MTPConfig(assistant_model_path=args.official)
    draft = Gemma4MTPDraftModel(cfg).to(dev).eval()
    draft.load_assistant_weights(args.official)
    model = Gemma4MTPModel(
        draft_model=draft, mtp_num_steps=args.k,
        loss_decay_gamma=7.0, teacher_force=args.teacher_force,
    ).to(dev).eval()

    K = args.k
    acc_num = torch.zeros(K, device=dev)   # sum(acc*count) per step
    acc_den = torch.zeros(K, device=dev)   # sum(count) per step
    for i, tgt_ids in enumerate(seqs):
        attn = torch.ones_like(tgt_ids)
        with torch.no_grad():
            out = inner(input_ids=tgt_ids, attention_mask=attn,
                        use_cache=True, return_shared_kv_states=True)
            last_hidden = out.last_hidden_state.to(torch.bfloat16)
            shared_kv = out.shared_kv_states
            loss_mask = torch.ones_like(tgt_ids)
            loss, acc, loss_ps, acc_ps, cnt_ps = model(
                input_ids=tgt_ids,
                target_last_hidden=last_hidden,
                shared_kv_states=shared_kv,
                loss_mask=loss_mask,
                target_embed_weight=embed_w,
                target_lm_head_weight=lm_head_w,
            )
        for k in range(K):
            acc_num[k] += acc_ps[k] * cnt_ps[k]
            acc_den[k] += cnt_ps[k]
        print(f"  prompt {i} (T={tgt_ids.shape[1]}): "
              f"step0_acc={acc_ps[0].item():.4f}")

    print("=" * 60)
    print(f"OFFICIAL weights through TRAINING forward "
          f"(teacher_force={args.teacher_force}, {len(seqs)} prompts)")
    for k in range(K):
        a = (acc_num[k] / acc_den[k].clamp_min(1)).item()
        print(f"  step {k}: acc={a:.4f}  (count={int(acc_den[k].item())})")
    print("-" * 60)
    print("Expect step-0 acc HIGH (~0.8+) if training forward is aligned with")
    print("vLLM (official draft is known-good). Low pos0 => alignment still wrong.")
    print("=" * 60)


if __name__ == "__main__":
    main()
