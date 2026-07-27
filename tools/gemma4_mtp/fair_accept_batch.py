#!/usr/bin/env python
"""
FAIR per-position accept: HF engine vs vLLM engine, SAME prompts, N samples.

Reads the SAME sc1 prompt file vLLM bench reads (the "prompt" field, already
folded system+user -- byte-identical to what vLLM feeds), runs the full
speculative loop on the HF engine with the SAME rejection rule vLLM uses, and
reports per-position accept over N prompts. Compare against vLLM bench run on
the SAME file with --disable-shuffle and the same N.

Fairness guarantees:
  - identical prompt text (same file, same "prompt" field, no re-folding)
  - first N lines, no shuffle
  - only system+user is in the prompt; the model GENERATES (assistant labels
    are never fed) -- matches how vLLM bench measures accept
  - per-position accept uses the vLLM convention: denominator = number of drafts
    that REACHED position i (i.e. positions 0..i-1 were accepted so a draft at i
    was actually proposed). This is inherent to spec-decode (a draft at pos i
    only exists if the chain got there); vLLM reports the same conditional shape.

USAGE (server, 1 GPU), TorchSpec root:
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/fair_accept_batch.py \
      --target    /tmp/models/gemma4/text_only \
      --assistant /tmp/models/gemma4/assistant \
      --sc1 /path/to/vllm-msn/benchmarks/gemma4_12b_fp8/maiprofile_bench_prompts/sc1_maiprofile_layer1_delta.jsonl \
      --num-samples 200 --k 5 --temperature 0.7 --max-new 128 --seed 0
Run once per model (official vs trained), compare pos0..pos4 against the vLLM
bench per-position accept on the SAME sc1 file (NUM_PROMPTS=200, --disable-shuffle).
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F


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
    ap.add_argument("--assistant", required=True)
    ap.add_argument("--sc1", required=True, help="the SAME sc1 prompt file vLLM bench reads")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Gemma4AssistantForCausalLM,
    )

    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(dev).eval()
    backbone = find_backbone(target)
    tgt_embed = backbone.embed_tokens.weight
    embed_scale = torch.tensor(tgt_embed.shape[1] ** 0.5, dtype=tgt_embed.dtype, device=dev)
    draft = Gemma4AssistantForCausalLM.from_pretrained(
        args.assistant, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(dev).eval()

    # Read the SAME prompts vLLM reads: the "prompt" field, first N, no shuffle.
    prompts = []
    with open(args.sc1, encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= args.num_samples:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec.get("prompt")
            if isinstance(p, str) and p:
                prompts.append(p)
    print(f"loaded {len(prompts)} prompts from {args.sc1.split('/')[-1]}  "
          f"k={args.k} temp={args.temperature} max_new={args.max_new}")

    temp = max(args.temperature, 1e-6)
    greedy = args.temperature == 0.0

    def target_forward(full_ids):
        with torch.no_grad():
            o = backbone(full_ids, use_cache=True, return_shared_kv_states=True,
                         output_hidden_states=False)
        return o.last_hidden_state, o.shared_kv_states

    def tl_from_h(h):
        return (h.float() @ tgt_embed.float().T)

    accept_hits = [0] * args.k
    accept_tot = [0] * args.k

    for si, ptext in enumerate(prompts):
        ids = tok(ptext, return_tensors="pt", truncation=True, max_length=args.max_len).input_ids.to(dev)
        cur = ids
        generated = 0
        while generated < args.max_new:
            t_hidden, shared_kv = target_forward(cur)
            prev_hidden = t_hidden[:, -1:, :]
            last_tok = cur[:, -1:]
            draft_tokens, draft_logits_list = [], []
            for j in range(args.k):
                tok_emb = F.embedding(last_tok.clamp(0, tgt_embed.shape[0]-1), tgt_embed) * embed_scale
                inp = torch.cat([tok_emb, prev_hidden], dim=-1)
                pos = torch.tensor([[cur.shape[1] - 1 + j]], device=dev)
                with torch.no_grad():
                    out = draft(inputs_embeds=inp, position_ids=pos, shared_kv_states=shared_kv)
                dl = out.logits[:, -1, :]
                draft_logits_list.append(dl)
                nt = dl.argmax(-1, keepdim=True) if greedy else torch.multinomial(F.softmax(dl/temp, -1), 1)
                draft_tokens.append(nt)
                prev_hidden = out.last_hidden_state[:, -1:, :]
                last_tok = nt
            prop = torch.cat(draft_tokens, dim=1)
            verify_ids = torch.cat([cur, prop], dim=1)
            vh, _ = target_forward(verify_ids)
            L = cur.shape[1]
            n_acc = 0
            for i in range(args.k):
                th = vh[:, L - 1 + i, :]
                tl = tl_from_h(th)
                dtok = prop[0, i].item()
                accept_tot[i] += 1
                if greedy:
                    acc = (tl.argmax(-1).item() == dtok)
                else:
                    pt = F.softmax(tl/temp, -1)[0, dtok].item()
                    pd = F.softmax(draft_logits_list[i]/temp, -1)[0, dtok].item()
                    acc = (torch.rand(1).item() < min(1.0, pt/max(pd, 1e-12)))
                if acc:
                    accept_hits[i] += 1
                    n_acc += 1
                else:
                    break
            corr_h = vh[:, L - 1 + min(n_acc, args.k - 1), :]
            corr_l = tl_from_h(corr_h)
            corr = corr_l.argmax(-1, keepdim=True) if greedy else torch.multinomial(F.softmax(corr_l/temp, -1), 1)
            cur = torch.cat([verify_ids[:, : L + n_acc], corr], dim=1)
            generated += n_acc + 1
            if corr.item() == tok.eos_token_id:
                break
        if (si + 1) % 20 == 0:
            print(f"  ...{si+1}/{len(prompts)} done", flush=True)

    print("\n============ FAIR per-position accept (HF engine) ============")
    print(f"model: {args.assistant.rstrip('/').split('/')[-1]}  prompts={len(prompts)}")
    for i in range(args.k):
        r = accept_hits[i] / max(accept_tot[i], 1)
        print(f"  Position {i}: {100*r:6.2f}%   ({accept_hits[i]}/{accept_tot[i]})")
    overall = sum(accept_hits) / max(sum(accept_tot), 1)
    print(f"  overall accept: {100*overall:.2f}%   acc_len~{sum(accept_hits)/max(accept_tot[0],1):.2f}")
    print("--------------------------------------------------------------")
    print("Compare pos0..pos4 vs vLLM bench on the SAME sc1 file")
    print("(NUM_PROMPTS={}, add --disable-shuffle to the bench call).".format(len(prompts)))
    print("==============================================================")


if __name__ == "__main__":
    main()
