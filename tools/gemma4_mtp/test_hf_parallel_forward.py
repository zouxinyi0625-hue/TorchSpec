#!/usr/bin/env python
"""
Test whether the HF Gemma4Assistant PARALLEL forward (create_attention_masks path)
matches vLLM at the sampled position, using the exact verified inputs from the
strip dump. This decides "Option Y" feasibility: if HF parallel forward reproduces
vLLM's draft argmax at position s, training can keep using Gemma4AssistantForCausalLM
directly; if not, we must port the strip single-step logic (Option X).

Inputs (all from /tmp/vllm_draft_step0.pt, verified correct by strip):
  - input_ids        (N,)      draft input tokens (shifted: input_ids[t]=token_{t+1})
  - hidden_states    (N,2816)  target hidden fed to the draft
  - tgt_fullkv       target's real full-layer K/V (known-correct, argmax matched)
  - shared_kv_gathered  L28 sliding K/V (gather correct for sliding)
  - token_indices_to_sample / draft_token_ids  vLLM's sampled pos + argmax

We feed HF the full sequence in ONE parallel call (q_len=N) and read logits at s.

USAGE (server, GPU + transformers 5.9):
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/test_hf_parallel_forward.py \
      --official $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
      --target   $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
      --dump /tmp/vllm_draft_step0.pt
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def find_embed(m):
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
    ap.add_argument("--dump", default="/tmp/vllm_draft_step0.pt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, Gemma4AssistantForCausalLM

    dev = args.device
    d = torch.load(args.dump, map_location="cpu")
    input_ids = d["input_ids"].to(dev)                        # (N,)
    hidden = d["hidden_states"].to(dev).to(torch.bfloat16)    # (N,2816)
    tis = d["token_indices_to_sample"].flatten().tolist()
    vllm_draft = d["draft_token_ids"].flatten().tolist()
    N = input_ids.shape[0]
    s = tis[0]
    print(f"N={N} sampled={s} vllm_draft={vllm_draft}")

    # Build shared_kv_states in HF's contract: {layer_type: (K, V)} shaped
    # (B, heads, kv_len, dim). Use the KNOWN-CORRECT tensors:
    #   full_attention <- tgt_fullkv (target's real full-layer K/V, verified)
    #   sliding_attention <- shared_kv_gathered L28 (verified correct for sliding)
    tf = d["tgt_fullkv"][-1]
    kf = tf["k"].to(dev).to(torch.bfloat16)                   # (Nt,1024)
    vf = tf["v"].to(dev).to(torch.bfloat16)
    nkvh_f, hd_f = tf["num_kv_heads"], tf["head_dim"]         # 2, 512
    kf = kf.view(kf.shape[0], nkvh_f, hd_f).permute(1, 0, 2).unsqueeze(0)  # (1,2,Nt,512)
    vf = vf.view(vf.shape[0], nkvh_f, hd_f).permute(1, 0, 2).unsqueeze(0)
    full_kv = (kf, vf)

    g = d["shared_kv_gathered"]
    sl_key = [k for k in g if "28" in k][0]
    ks = g[sl_key]["k"].to(dev).to(torch.bfloat16)            # (N,8,256)
    vs = g[sl_key]["v"].to(dev).to(torch.bfloat16)
    ks = ks.permute(1, 0, 2).unsqueeze(0)                     # (1,8,N,256)
    vs = vs.permute(1, 0, 2).unsqueeze(0)
    sliding_kv = (ks, vs)

    shared_kv = {"full_attention": full_kv, "sliding_attention": sliding_kv}
    print(f"shared_kv full={tuple(full_kv[0].shape)} sliding={tuple(sliding_kv[0].shape)}")

    # inputs_embeds = cat[ target_embed(input_ids)*sqrt(2816), hidden ]
    print("loading target embed ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, trust_remote_code=True).eval()
    embed_w = find_embed(target).to(dev)
    scale = float(hidden.shape[-1] ** 0.5)
    tok_emb = F.embedding(input_ids, embed_w) * scale
    inputs_embeds = torch.cat([tok_emb, hidden], dim=-1).unsqueeze(0)  # (1,N,5632)
    del target

    print("loading draft ...")
    draft = Gemma4AssistantForCausalLM.from_pretrained(
        args.official, dtype=torch.bfloat16).to(dev).eval()
    pos = torch.arange(N, device=dev).unsqueeze(0)

    with torch.no_grad():
        out = draft(inputs_embeds=inputs_embeds, position_ids=pos,
                    shared_kv_states=shared_kv)
        argmax_s = int(out.logits[0, s].argmax().item())
        top5 = out.logits[0, s].topk(5).indices.tolist()

    print("=" * 60)
    print(f"HF parallel forward argmax @pos {s}: {argmax_s}")
    print(f"top5: {top5}")
    print(f"vLLM wants: {vllm_draft}")
    print("-" * 60)
    if argmax_s == vllm_draft[0]:
        print("=> MATCH: HF parallel forward == vLLM. Option Y works; training can")
        print("   keep Gemma4AssistantForCausalLM. Just fix token off-by-one + label.")
    else:
        print("=> DIFF: HF parallel forward != vLLM. The parallel sliding-window mask")
        print("   does not match vLLM. Use Option X (port strip single-step logic).")
    print("=" * 60)


if __name__ == "__main__":
    main()
