#!/usr/bin/env python
"""
Path A: HF bit-level parity check for the vLLM draft step-0 forward.

Loads the EXACT tensors vLLM fed/produced (from VLLM_DUMP_DRAFT_TENSORS):
  input_ids       (N,)         draft input token ids (already shifted by vLLM)
  hidden_states   (N, 2816)    target hidden fed to the draft
  target_token_ids(N,)         original target sequence (for recomputing shared_kv)
  draft_token_ids (M,)         vLLM's draft argmax at the sampled positions
  token_indices_to_sample      positions vLLM sampled the draft at

We recompute shared_kv from the TARGET on target_token_ids (bf16, no FP8), then
run the HF Gemma4Assistant draft on the EXACT input_ids + hidden_states vLLM used.
If HF's draft argmax at the sampled position == vLLM's draft_token_ids, the HF
forward reproduces vLLM (probe mechanism is valid). If not, the HF forward
differs (engine mismatch) -- with identical inputs, the only variable is the
forward implementation (and the shared_kv we recompute here).

USAGE:
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/hf_replay_draft_step0.py \
      --target   $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
      --official $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
      --dump /tmp/vllm_draft_step0.pt
"""
from __future__ import annotations

import argparse

import torch


def find_backbone(m):
    import torch.nn as nn
    q, seen = [m], set()
    while q:
        mod = q.pop(0)
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        if (hasattr(mod, "norm") and isinstance(getattr(mod, "norm"), nn.Module)
                and hasattr(mod, "layers")):
            return mod
        for _n, c in mod.named_children():
            q.append(c)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--dump", default="/tmp/vllm_draft_step0.pt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, Gemma4AssistantForCausalLM

    dev = args.device
    d = torch.load(args.dump, map_location="cpu")
    input_ids = d["input_ids"].to(dev)                 # (N,) draft input (shifted)
    hidden_states = d["hidden_states"].to(dev).to(torch.bfloat16)  # (N,2816)
    target_token_ids = d["target_token_ids"].to(dev)   # (N,) original target seq
    vllm_draft = d["draft_token_ids"].flatten().tolist()
    tis = d["token_indices_to_sample"]
    tis = tis.flatten().tolist() if hasattr(tis, "flatten") else tis
    N = input_ids.shape[0]
    print(f"loaded dump: N={N} sample_idx={tis} vllm_draft_argmax={vllm_draft}")

    # --- target: recompute shared_kv on the ORIGINAL target sequence (bf16) ---
    print("loading target (bf16) ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, trust_remote_code=True
    ).to(dev).eval()
    backbone = find_backbone(target)
    with torch.no_grad():
        tout = backbone(target_token_ids.unsqueeze(0), use_cache=True,
                        return_shared_kv_states=True)
        shared_kv = tout.shared_kv_states
        hf_target_hidden = tout.last_hidden_state[0]    # (N,2816) HF's own hidden
    # sanity: does vLLM's dumped hidden match HF's target hidden?
    cos = torch.nn.functional.cosine_similarity(
        hidden_states.float(), hf_target_hidden.float(), dim=-1)
    print(f"[hidden check] vLLM hidden vs HF target hidden: "
          f"cos mean={cos.mean():.4f} min={cos.min():.4f}")

    # --- draft: run on the EXACT vLLM input_ids + hidden_states ---
    print("loading official draft (bf16) ...")
    draft = Gemma4AssistantForCausalLM.from_pretrained(
        args.official, dtype=torch.bfloat16).to(dev).eval()
    backbone_hidden = hidden_states.shape[-1]
    scale = float(backbone_hidden ** 0.5)
    embed_w = backbone.embed_tokens.weight

    pos = torch.arange(N, device=dev).unsqueeze(0)
    tok_emb = torch.nn.functional.embedding(input_ids, embed_w) * scale  # (N,2816)
    inputs_embeds = torch.cat([tok_emb.unsqueeze(0),
                               hidden_states.unsqueeze(0)], dim=-1)  # (1,N,5632)

    with torch.no_grad():
        out = draft(inputs_embeds=inputs_embeds, position_ids=pos,
                    shared_kv_states=shared_kv)
        hf_draft_argmax = out.logits[0].argmax(-1)      # (N,)

    print("=" * 60)
    print("Draft argmax parity at vLLM's sampled positions:")
    match = 0
    for j, idx in enumerate(tis):
        hf_tok = int(hf_draft_argmax[idx].item())
        vl_tok = vllm_draft[j] if j < len(vllm_draft) else None
        ok = (hf_tok == vl_tok)
        match += int(ok)
        print(f"  pos {idx}: HF={hf_tok}  vLLM={vl_tok}  {'OK' if ok else 'DIFF'}")
    print("-" * 60)
    print(f"match {match}/{len(tis)}")
    if match == len(tis):
        print("=> HF forward REPRODUCES vLLM draft. Forward is equivalent;")
        print("   probe mechanism is valid, shared_kv recompute matches.")
    else:
        print("=> HF forward DIFFERS from vLLM. With identical input_ids+hidden,")
        print("   the gap is the forward impl and/or shared_kv. Need Path B")
        print("   (dump vLLM's actual shared_kv) to fully isolate.")
    print("=" * 60)


if __name__ == "__main__":
    main()
