#!/usr/bin/env python
"""
FAIR per-position accept comparison: same model, same sample, HF engine vs vLLM.

Runs the FULL speculative-decoding loop on the HF engine (draft proposes K
tokens autoregressively, target verifies with the SAME rejection rule vLLM
uses), and reports per-position accept rate. Compare directly against the vLLM
bench per-position accept for the same model + same prompt.

Why: if even the UNTRAINED official assistant does not match between HF and vLLM
per-position (esp. pos0, which has no prefix dependence), the gap is a
HF-vs-vLLM engine discrepancy, NOT a training problem -- retraining wouldn't
help. This isolates that.

Rejection rule (matches vLLM):
  - greedy (temperature==0):   accept iff draft_token == argmax(target_logits)
  - random (temperature>0):    accept iff u < min(1, p_target(t)/p_draft(t)),
                               u ~ Uniform(0,1); on reject, stop this draft.
Both draft and target logits are temperature-scaled before forming probs.

USAGE (server, 1 GPU), TorchSpec root:
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/fair_accept_hf_vs_vllm.py \
      --target    /tmp/models/gemma4/text_only \
      --assistant /tmp/models/gemma4/assistant \
      --data      data/eval_layer1_delta_1000.jsonl \
      --k 5 --temperature 0.7 --max-new 128 --seed 0 --sample-index 0
Then compare printed pos0..pos4 accept vs the vLLM bench per-position accept
for the SAME assistant + prompt.
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
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=5, help="draft tokens per step")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-index", type=int, default=0)
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

    # Build the prompt (system+user, drop assistant) exactly like the bench.
    with open(args.data, encoding="utf-8") as f:
        for _ in range(args.sample_index + 1):
            line = f.readline()
    rec = json.loads(line)
    convs = rec["conversations"]
    sys_c = next((t["content"] for t in convs if t["role"] == "system"), "")
    usr_c = next((t["content"] for t in convs if t["role"] == "user"), "")
    text = (sys_c + "\n\n" + usr_c) if sys_c else usr_c
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    prompt_len = ids.shape[1]
    print(f"prompt tokens: {prompt_len}  k={args.k}  temp={args.temperature}")

    temp = max(args.temperature, 1e-6)
    greedy = args.temperature == 0.0

    def target_forward(full_ids):
        with torch.no_grad():
            o = backbone(full_ids, use_cache=True, return_shared_kv_states=True,
                         output_hidden_states=False)
        return o.last_hidden_state, o.shared_kv_states

    def target_logits_from_hidden(h):
        return (h.float() @ tgt_embed.float().T)

    accept_hits = [0] * args.k
    accept_tot = [0] * args.k
    cur = ids
    steps = 0
    generated = 0

    while generated < args.max_new:
        steps += 1
        # 1. target forward on the current sequence -> hidden + shared_kv + next dist
        t_hidden, shared_kv = target_forward(cur)
        # last position's target distribution (the bonus / first-token dist)
        # 2. draft proposes k tokens autoregressively, seeded by target's last hidden
        prev_hidden = t_hidden[:, -1:, :]                      # (1,1,2816)
        last_tok = cur[:, -1:]                                 # (1,1)
        draft_tokens = []
        draft_logits_list = []
        for j in range(args.k):
            tok_emb = F.embedding(last_tok.clamp(0, tgt_embed.shape[0]-1), tgt_embed) * embed_scale
            inp = torch.cat([tok_emb, prev_hidden], dim=-1)   # (1,1,5632)
            pos = torch.tensor([[cur.shape[1] - 1 + j]], device=dev)
            with torch.no_grad():
                out = draft(inputs_embeds=inp, position_ids=pos, shared_kv_states=shared_kv)
            dl = out.logits[:, -1, :]                          # (1,V)
            draft_logits_list.append(dl)
            if greedy:
                nt = dl.argmax(-1, keepdim=True)
            else:
                p = F.softmax(dl / temp, dim=-1)
                nt = torch.multinomial(p, 1)
            draft_tokens.append(nt)
            prev_hidden = out.last_hidden_state[:, -1:, :]
            last_tok = nt

        # 3. target verifies: forward the proposed tokens appended, get target dist
        #    at each proposed position, apply the rejection rule.
        prop = torch.cat(draft_tokens, dim=1)                 # (1,k)
        verify_ids = torch.cat([cur, prop], dim=1)
        vh, _ = target_forward(verify_ids)
        # target dist that predicts position (prompt_len-1+i) is at hidden[..-k-1+i]
        # positions of the k proposed tokens inside verify_ids: [L, L+1, ..., L+k-1]
        L = cur.shape[1]
        n_accept_this = 0
        for i in range(args.k):
            # target distribution predicting the i-th proposed token comes from
            # hidden at index (L-1+i) in verify_ids.
            th = vh[:, L - 1 + i, :]                           # (1,2816)
            tl = target_logits_from_hidden(th)                # (1,V)
            dtok = prop[0, i].item()
            accept_tot[i] += 1
            if greedy:
                acc = (tl.argmax(-1).item() == dtok)
            else:
                pt = F.softmax(tl / temp, dim=-1)[0, dtok].item()
                pd = F.softmax(draft_logits_list[i] / temp, dim=-1)[0, dtok].item()
                ratio = min(1.0, pt / max(pd, 1e-12))
                u = torch.rand(1).item()
                acc = (u < ratio)
            if acc:
                accept_hits[i] += 1
                n_accept_this += 1
            else:
                break  # first rejection stops the chain (standard spec-decode)

        # advance: accepted tokens + 1 bonus/correction token
        n_take = n_accept_this + 1
        cur = verify_ids[:, : L + min(n_accept_this, args.k)]
        # append one target-sampled correction token from the first rejected pos
        corr_h = vh[:, L - 1 + min(n_accept_this, args.k - 1), :]
        corr_l = target_logits_from_hidden(corr_h)
        if greedy:
            corr = corr_l.argmax(-1, keepdim=True)
        else:
            corr = torch.multinomial(F.softmax(corr_l / temp, dim=-1), 1)
        cur = torch.cat([cur, corr], dim=1)
        generated += n_take
        if corr.item() == tok.eos_token_id:
            break

    print("\n============ FAIR per-position accept (HF engine) ============")
    print(f"model: {args.assistant.split('/')[-1]}  steps={steps}  generated~{generated}")
    for i in range(args.k):
        r = accept_hits[i] / max(accept_tot[i], 1)
        print(f"  Position {i}: {100*r:6.2f}%   ({accept_hits[i]}/{accept_tot[i]})")
    overall = sum(accept_hits) / max(sum(accept_tot), 1)
    print(f"  overall accept: {100*overall:.2f}%")
    print("--------------------------------------------------------------")
    print("Compare these pos0..pos{} vs the vLLM bench per-position accept for the".format(args.k-1))
    print("SAME assistant + same prompt. If pos0 differs a lot, it's an HF-vs-vLLM")
    print("engine discrepancy (not training). Note: small sample -> run more steps")
    print("via --max-new and average across --sample-index for a stable number.")
    print("==============================================================")


if __name__ == "__main__":
    main()
