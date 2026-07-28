#!/usr/bin/env python
"""
Path B closure: reproduce vLLM's draft step-0 argmax OUTSIDE vLLM, using the
EXACT tensors vLLM fed/produced (VLLM_DUMP_DRAFT_TENSORS dump), with the real
target shared_kv gathered from the paged cache (no recompute).

vLLM's draft step-0 (llm_base_proposer.propose):
  - sample_hidden_states = last_hidden_states[token_indices_to_sample]  (q_len=1)
  - the draft consumed input_ids (shifted: input_ids[t]=token_{t+1}) + hidden_t
  - attention reads K/V from the target cache (our gathered shared_kv)
  - draft_token_ids = argmax(lm_head(sample_hidden))

We load those exact inputs and run the HF Gemma4Assistant draft, feeding the
gathered shared_kv (reshaped to HF's {layer_type:(K,V)} contract), then compare
the draft argmax at the sampled position against vLLM's dumped draft_token_ids.

  MATCH  => the forward is reproducible outside vLLM with identical inputs.
            The training forward can be aligned to this exactly => train==infer.
  DIFF   => HF Gemma4Assistant forward != vLLM gemma4_mtp forward (impl gap);
            we then port vLLM's gemma4_mtp math directly.

USAGE:
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/replay_vllm_draft_step0.py \
      --official $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
      --target   $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
      --dump /tmp/vllm_draft_step0.pt
"""
from __future__ import annotations

import argparse

import torch


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
    input_ids = d["input_ids"].to(dev)                       # (N,) draft input (shifted)
    hidden = d["hidden_states"].to(dev).to(torch.bfloat16)   # (N,2816) target hidden fed to draft
    tis = d["token_indices_to_sample"].flatten().tolist()    # sampled positions
    vllm_draft = d["draft_token_ids"].flatten().tolist()
    gathered = d["shared_kv_gathered"]                       # {tgt_layer: {k,v}}
    N = input_ids.shape[0]
    print(f"N={N} sampled_positions={tis} vllm_draft_argmax={vllm_draft}")
    for k, v in gathered.items():
        print(f"  gathered {k}: k={tuple(v['k'].shape)} v={tuple(v['v'].shape)}")

    # Map gathered target layers -> HF shared_kv_states contract.
    # L28 = sliding_attention source, L29 = full_attention source (Gemma4).
    # HF expects {layer_type: (K, V)} with K,V shaped (B, heads, kv_len, dim).
    def to_hf_kv(entry):
        k = entry["k"].to(dev).to(torch.bfloat16)   # (N, heads, dim)
        v = entry["v"].to(dev).to(torch.bfloat16)
        # (N,h,d) -> (1,h,N,d)
        return (k.permute(1, 0, 2).unsqueeze(0).contiguous(),
                v.permute(1, 0, 2).unsqueeze(0).contiguous())

    keys = list(gathered.keys())
    sliding_key = [k for k in keys if "28" in k]
    full_key = [k for k in keys if "29" in k]
    shared_kv = {}
    if sliding_key:
        shared_kv["sliding_attention"] = to_hf_kv(gathered[sliding_key[0]])
    if full_key:
        shared_kv["full_attention"] = to_hf_kv(gathered[full_key[0]])
    print("shared_kv keys:", list(shared_kv.keys()))

    # Build inputs_embeds = cat[ embed(input_ids)*sqrt(2816), hidden ] using the
    # TARGET embedding table (the draft's embed_tokens is replaced by target's).
    print("loading target embed table ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, trust_remote_code=True).eval()
    import torch.nn as nn
    q = [target]
    embed_w = None
    while q:
        m = q.pop(0)
        if isinstance(m, nn.Module) and hasattr(m, "embed_tokens"):
            embed_w = m.embed_tokens.weight
            break
        q.extend(list(m.children()))
    scale = float(hidden.shape[-1] ** 0.5)
    tok_emb = torch.nn.functional.embedding(input_ids, embed_w.to(dev)) * scale
    inputs_embeds = torch.cat([tok_emb.unsqueeze(0), hidden.unsqueeze(0)], dim=-1)
    del target

    print("loading official draft ...")
    draft = Gemma4AssistantForCausalLM.from_pretrained(
        args.official, dtype=torch.bfloat16).to(dev).eval()
    pos = torch.arange(N, device=dev).unsqueeze(0)

    with torch.no_grad():
        out = draft(inputs_embeds=inputs_embeds, position_ids=pos,
                    shared_kv_states=shared_kv)
        hf_argmax = out.logits[0].argmax(-1)     # (N,)

    print("=" * 60)
    match = 0
    for j, idx in enumerate(tis):
        hf_tok = int(hf_argmax[idx].item())
        vl_tok = vllm_draft[j] if j < len(vllm_draft) else None
        ok = hf_tok == vl_tok
        match += int(ok)
        print(f"  pos {idx}: HF={hf_tok}  vLLM={vl_tok}  {'OK' if ok else 'DIFF'}")
    print("-" * 60)
    print(f"match {match}/{len(tis)}")
    if match == len(tis):
        print("=> REPRODUCED vLLM draft outside vLLM with identical inputs+shared_kv.")
        print("   Forward is portable; align training to this exact math => train==infer.")
    else:
        print("=> DIFF: HF Gemma4Assistant forward != vLLM gemma4_mtp forward.")
        print("   Same input_ids+hidden+shared_kv, different argmax -> port vLLM math.")
    print("=" * 60)


if __name__ == "__main__":
    main()
