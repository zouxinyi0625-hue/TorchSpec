#!/usr/bin/env python
"""
Standalone single-step replay of vLLM's Gemma4 MTP draft, OUTSIDE vLLM.

Instead of calling the HF model's forward (which builds its own shared-kv
attention mask that does not match vLLM's single-step decode), we drive the HF
draft's own submodules (pre_projection, per-layer q_proj/q_norm/norms/mlp,
final norm, lm_head) by hand and do the attention as vLLM does:

  * only the SAMPLED position is a query (q_len = 1)
  * that query attends over ALL gathered target K/V (the real shared_kv from the
    paged cache, already post-norm + post-RoPE)
  * scaling = 1.0 (Gemma4 MTP), no causal mask needed for a single query
  * K/V are the vLLM-gathered tensors -> no recompute, no HF mask

If argmax == vLLM's dumped draft token (188357), the draft forward is
reproducible outside vLLM and can be ported into training for train==infer.

USAGE (server, needs GPU + torch + transformers 5.9):
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/strip_vllm_draft_step0.py \
      --official $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
      --target   $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
      --dump /tmp/vllm_draft_step0.pt
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def find_module_with(m, attr):
    q = [m]
    while q:
        x = q.pop(0)
        if hasattr(x, attr):
            return getattr(x, attr)
        q.extend(list(x.children()))
    return None


def find_layers(m):
    q = [m]
    while q:
        x = q.pop(0)
        if hasattr(x, "layers") and isinstance(getattr(x, "layers"), nn.ModuleList):
            return x.layers
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
    input_ids = d["input_ids"].to(dev)                       # (N,)
    hidden = d["hidden_states"].to(dev).to(torch.bfloat16)   # (N,2816)
    positions = d["positions"].to(dev).flatten()             # (N,)
    tis = d["token_indices_to_sample"].flatten().tolist()
    vllm_draft = d["draft_token_ids"].flatten().tolist()
    gathered = d["shared_kv_gathered"]
    N = input_ids.shape[0]
    s = tis[0]                                               # sampled position (4368)
    print(f"N={N} sampled={s} vllm_draft={vllm_draft} pos[s]={int(positions[s])}")

    # gathered K/V: (N, heads, dim), post-norm + post-RoPE (target's)
    kv = {}
    for k, v in gathered.items():
        lt = "sliding_attention" if "28" in k else "full_attention"
        kv[lt] = (v["k"].to(dev).to(torch.bfloat16), v["v"].to(dev).to(torch.bfloat16))
        print(f"  {lt}: k={tuple(kv[lt][0].shape)} v={tuple(kv[lt][1].shape)}")

    # --- target embed table (draft's embed_input_ids uses target's, *sqrt(2816)) ---
    print("loading target embed ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, trust_remote_code=True).eval()
    embed_w = find_module_with(target, "embed_tokens").weight.to(dev)
    scale = float(hidden.shape[-1] ** 0.5)                   # sqrt(2816)

    print("loading draft ...")
    draft = Gemma4AssistantForCausalLM.from_pretrained(
        args.official, dtype=torch.bfloat16).to(dev).eval()
    layers = find_layers(draft)
    pre_projection = find_module_with(draft, "pre_projection")
    final_norm = None
    # the model-level final norm (not a layer's); grab draft.model.norm
    dm = find_module_with(draft, "layers")  # this is ModuleList; need parent
    # locate parent module holding .layers and .norm
    parent = None
    q = [draft]
    while q:
        x = q.pop(0)
        if hasattr(x, "layers") and hasattr(x, "norm"):
            parent = x
            break
        q.extend(list(x.children()))
    final_norm = parent.norm
    lm_head = draft.get_output_embeddings()

    # rotary embeddings: reuse each layer's own rotary if present, else the
    # model-level rotary_emb. Gemma4 computes position_embeddings once and
    # passes (cos,sin) to layers. We need per-layer-type rope (sliding/full).
    rotary = find_module_with(draft, "rotary_emb")

    with torch.no_grad():
        # inputs_embeds = cat[ embed(input_ids)*scale , hidden ]  (draft path)
        tok_emb = F.embedding(input_ids, embed_w) * scale        # (N,2816)
        combined = torch.cat([tok_emb, hidden], dim=-1)          # (N,5632)
        h = pre_projection(combined)                             # (N, H)
        H = h.shape[-1]

        # position embeddings for all N (needed for K side already applied, but
        # we only rope the single query; use per-layer rotary if it differs).
        pos_ids = positions.unsqueeze(0)                        # (1,N)

        for li, layer in enumerate(layers):
            attn = layer.self_attn
            lt = attn.layer_type
            hd = attn.head_dim
            nh = attn.config.num_attention_heads
            residual = h
            x = layer.input_layernorm(h)                        # (N,H)

            # q for the sampled position only
            q = attn.q_proj(x[s:s + 1])                          # (1, nh*hd)
            q = q.view(1, nh, hd)
            q = attn.q_norm(q)                                  # (1,nh,hd)
            # rope on q at position pos[s]
            cos, sin = rotary(x[s:s + 1].unsqueeze(0), pos_ids[:, s:s + 1],
                              layer_type=lt)
            # apply_rotary: q (1,nh,hd), cos/sin (1,1,hd)
            from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
            q_r = apply_rotary_pos_emb(q.unsqueeze(0), cos, sin, unsqueeze_dim=2)[0]  # (1,nh,hd)?
            q_r = q_r.reshape(1, nh, hd).transpose(0, 1)        # (nh,1,hd)

            k, v = kv[lt]                                       # (N, kvh, hd)
            kvh = k.shape[1]
            k_t = k.transpose(0, 1)                             # (kvh, N, hd)
            v_t = v.transpose(0, 1)
            # GQA: repeat kv heads to nh
            rep = nh // kvh
            k_t = k_t.repeat_interleave(rep, dim=0)             # (nh,N,hd)
            v_t = v_t.repeat_interleave(rep, dim=0)
            # single-query attention over all N, scaling=1.0, no mask
            attn_w = torch.matmul(q_r, k_t.transpose(-1, -2)) * 1.0   # (nh,1,N)
            attn_w = attn_w.softmax(dim=-1)
            attn_o = torch.matmul(attn_w, v_t)                 # (nh,1,hd)
            attn_o = attn_o.transpose(0, 1).reshape(1, nh * hd)
            attn_o = attn.o_proj(attn_o)                       # (1,H)

            # write back only sampled position; other positions unchanged (we
            # only need the sampled position's final logit)
            hs = layer.post_attention_layernorm(attn_o) + residual[s:s + 1]
            r2 = hs
            ff = layer.pre_feedforward_layernorm(hs)
            ff = layer.mlp(ff)
            ff = layer.post_feedforward_layernorm(ff)
            hs = ff + r2
            # layer_scalar if present
            ls = getattr(layer, "layer_scalar", None)
            if ls is not None:
                hs = hs * ls
            # only the sampled row propagates; replace row s
            h = h.clone()
            h[s:s + 1] = hs

        draft_hidden = final_norm(h[s:s + 1])                  # (1,H)
        logits = lm_head(draft_hidden)                         # (1,V)
        argmax = int(logits[0].argmax().item())
        top5 = logits[0].topk(5).indices.tolist()

    # --- layered verification against vLLM dump ---
    print("--- layered check vs dump ---")
    vllm_sample_hidden = d.get("sample_hidden_states")
    if vllm_sample_hidden is not None:
        vsh = vllm_sample_hidden.to(dev).to(torch.float32)      # (1,1024) draft-dim
        mine = draft_hidden.to(torch.float32)
        cos = F.cosine_similarity(mine, vsh, dim=-1)
        print(f"  my draft_hidden vs vLLM sample_hidden: cos={cos.mean().item():.4f} "
              f"my_norm={mine.norm().item():.2f} vllm_norm={vsh.norm().item():.2f}")
        # what token does vLLM's OWN sample_hidden produce via this lm_head?
        vllm_logits = lm_head(vsh.to(torch.bfloat16))
        print(f"  lm_head(vLLM sample_hidden) argmax={int(vllm_logits[0].argmax())} "
              f"(should be {vllm_draft[0]})")

    print("=" * 60)
    print(f"stripped draft argmax @pos {s}: {argmax}")
    print(f"top5: {top5}")
    print(f"vLLM wants: {vllm_draft}")
    print("-" * 60)
    if argmax == vllm_draft[0]:
        print("=> MATCH: stripped vLLM draft forward reproduced outside vLLM.")
        print("   Port this exact math to training => train==infer.")
    elif vllm_draft[0] in top5:
        print("=> CLOSE: vLLM token in top5 but not top1; small numerical gap,")
        print("   inspect rope/scaling/kv-head repeat.")
    else:
        print("=> DIFF: still off. Inspect embed scale, rope params (sliding vs")
        print("   full theta), or whether gathered K/V are pre/post rope.")
    print("=" * 60)


if __name__ == "__main__":
    main()
