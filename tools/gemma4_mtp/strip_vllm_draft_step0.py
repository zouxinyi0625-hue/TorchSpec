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
    ap.add_argument("--sliding-window", type=int, default=1024)
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

    # Build vLLM's OWN rope per layer-type (exact vLLM math, incl. partial
    # rotary for full attention). sliding: default theta=10000, full_dim=256;
    # full: proportional theta=1e6, partial_rotary_factor=0.25, head=512.
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.config import VllmConfig, set_current_vllm_config
    _vc = VllmConfig()
    with set_current_vllm_config(_vc):
        vllm_ropes = {
            "sliding_attention": get_rope(
                head_size=256, max_position=131072, is_neox_style=True,
                dtype=torch.bfloat16,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0}),
            "full_attention": get_rope(
                head_size=512, max_position=131072, is_neox_style=True,
                dtype=torch.bfloat16,
                rope_parameters={"rope_type": "proportional", "rope_theta": 1000000.0,
                                 "partial_rotary_factor": 0.25}),
        }
    for _rp in vllm_ropes.values():
        _rp.to(dev)

    with torch.no_grad():
        # inputs_embeds = cat[ embed(input_ids)*scale , hidden ]  (draft path)
        tok_emb = F.embedding(input_ids, embed_w) * scale        # (N,2816)
        combined = torch.cat([tok_emb, hidden], dim=-1)          # (N,5632)
        h = pre_projection(combined)                             # (N, H)
        H = h.shape[-1]

        # vLLM per-layer hidden for parity check
        vllm_layers = dict(d.get("layer_hidden") or [])
        def chk(name, mine_row):
            ref = vllm_layers.get(name)
            if ref is None:
                return
            r = ref[s].to(dev).to(torch.float32) if ref.dim() == 2 else ref.to(dev).to(torch.float32)
            m = mine_row.reshape(-1).to(torch.float32)
            c = F.cosine_similarity(m.unsqueeze(0), r.reshape(1, -1), dim=-1).item()
            print(f"  [parity {name}] cos={c:.4f} my_norm={m.norm():.2f} vllm_norm={r.norm():.2f}")
        chk("pre_projection", h[s])

        # position embeddings for all N (needed for K side already applied, but
        # we only rope the single query; use per-layer rotary if it differs).
        pos_ids = positions.unsqueeze(0)                        # (1,N)

        for li, layer in enumerate(layers):
            attn = layer.self_attn
            lt = attn.layer_type
            hd = attn.head_dim
            nh = attn.config.num_attention_heads
            if li == 3:
                qw = attn.q_proj.weight.shape
                qn_w = attn.q_norm.weight.shape if hasattr(attn.q_norm, "weight") else "?"
                kk, vv = kv[lt]
                kv_eq = torch.allclose(kk, vv)
                print(f"  [dbg L{li}] lt={lt} head_dim={hd} num_heads={nh} "
                      f"q_proj={tuple(qw)} q_out_per_head={qw[0]//nh} "
                      f"kv_shape={tuple(kv[lt][0].shape)} q_norm_w={qn_w} "
                      f"is_kv_shared={getattr(attn,'is_kv_shared_layer','?')} "
                      f"use_alt={getattr(attn,'use_alternative_attention','?')} "
                      f"k_eq_v={kv_eq} nkvg={nh//kk.shape[1]}")
            residual = h
            x = layer.input_layernorm(h)                        # (N,H)

            # q for the sampled position only
            q = attn.q_proj(x[s:s + 1])                          # (1, nh*hd)
            q = q.view(1, nh, hd)
            q = attn.q_norm(q)                                  # (1,nh,hd)
            # rope on q using vLLM's own rope (exact vLLM math, incl. partial
            # rotary for the full-attention layer). q flattened to (tokens, nh*hd).
            vrope = vllm_ropes[lt]
            q_flat = q.reshape(1, nh * hd)                       # (1, nh*hd)
            pos1 = positions[s:s + 1]                            # (1,)
            q_rot, _ = vrope.forward_native(pos1, q_flat, None)
            q_r = q_rot[0].view(nh, hd).unsqueeze(1)             # (nh,1,hd) - match USE_VLLM_Q path

            # isolate q_norm vs rope: compare my pre-rope q vs vLLM's pre-rope q
            if li == 3 and d.get("attn_dump") and d["attn_dump"][li].get("q_prerope") is not None:
                vqp = d["attn_dump"][li]["q_prerope"][s].to(dev).to(torch.float32)
                myqp = q_flat[0].to(torch.float32)
                cpre = F.cosine_similarity(myqp.unsqueeze(0), vqp.unsqueeze(0), dim=-1).item()
                vqpost = d["attn_dump"][li]["q_postrope"][s].to(dev).to(torch.float32)
                myqpost = q_rot[0].to(torch.float32)
                cpost = F.cosine_similarity(myqpost.unsqueeze(0), vqpost.unsqueeze(0), dim=-1).item()
                maxdiff = (myqpost - vqpost).abs().max().item()
                # per-head cos
                mh = myqpost.view(nh, hd); vh = vqpost.view(nh, hd)
                phc = F.cosine_similarity(mh, vh, dim=-1)
                print(f"  [q isolate L3] prerope_cos={cpre:.4f} postrope_cos={cpost:.4f} "
                      f"maxdiff={maxdiff:.4f} perhead_cos_min={phc.min():.4f} "
                      f"my_postnorm={myqpost.norm():.3f} vllm_postnorm={vqpost.norm():.3f}")

            # ISOLATION: optionally replace my q with vLLM's dumped q_postrope
            # to test whether the residual full-layer error is in q or attention.
            import os as _osq
            if _osq.environ.get("USE_VLLM_Q") == "1" and d.get("attn_dump") \
                    and li < len(d["attn_dump"]):
                vq = d["attn_dump"][li]["q_postrope"][s].to(dev).to(q_r.dtype)
                q_r = vq.view(nh, hd).unsqueeze(1)               # (nh,1,hd)

            k, v = kv[lt]                                       # (N, kvh, hd)
            kvh = k.shape[1]
            # compare gathered K/V vs target's real full-layer K/V (dump)
            if li == 3 and d.get("tgt_fullkv"):
                tk = d["tgt_fullkv"][-1]["k"].to(dev).to(torch.float32)  # (Nt,1024)
                tv = d["tgt_fullkv"][-1]["v"].to(dev).to(torch.float32)
                nmin = min(tk.shape[0], k.shape[0])
                # reshape target K to (Nt, heads, dim) to match gather layout
                tk_h = tk.view(tk.shape[0], kvh, hd)
                tv_h = tv.view(tv.shape[0], kvh, hd)
                # strict elementwise diff (not just global cos)
                kdiff = (k[:nmin].to(torch.float32) - tk_h[:nmin]).abs().max().item()
                vdiff = (v[:nmin].to(torch.float32) - tv_h[:nmin]).abs().max().item()
                # per-head cos at the sampled position
                ks = k[s].to(torch.float32); tks = tk_h[s]
                phk = F.cosine_similarity(ks, tks, dim=-1)
                print(f"  [gather vs target K/V] k_maxdiff={kdiff:.4f} v_maxdiff={vdiff:.4f} "
                      f"perhead_k_cos@s={[round(x,3) for x in phk.tolist()]} "
                      f"gather={tuple(k.shape)} target={tuple(tk.shape)}")
            k_t = k.transpose(0, 1)                             # (kvh, N, hd)
            v_t = v.transpose(0, 1)
            # GQA: repeat kv heads to nh
            rep = nh // kvh
            k_t = k_t.repeat_interleave(rep, dim=0)             # (nh,N,hd)
            v_t = v_t.repeat_interleave(rep, dim=0)
            # single-query attention, scaling=1.0
            attn_w = torch.matmul(q_r, k_t.transpose(-1, -2)) * 1.0   # (nh,1,N)

            # DECISIVE: for full layer, also compute attn with vLLM's dumped q,
            # using the SAME k_t/v_t, to see if the 0.06 q diff is the culprit.
            if li == 3 and d.get("attn_dump"):
                vq = d["attn_dump"][li]["q_postrope"][s].to(dev).to(q_r.dtype)
                vq_r = vq.view(nh, hd).unsqueeze(1)             # (nh,1,hd)
                print(f"  [shape L3] q_r={tuple(q_r.shape)} vq_raw={tuple(vq.shape)} "
                      f"vq_r={tuple(vq_r.shape)} k_t={tuple(k_t.shape)} v_t={tuple(v_t.shape)} "
                      f"attn_w={tuple(attn_w.shape)} nh={nh} hd={hd} kvh={kvh} rep={rep} N={N}")
                aw_v = torch.matmul(vq_r, k_t.transpose(-1, -2)).softmax(-1)
                ao_v = torch.matmul(aw_v, v_t).transpose(0, 1).reshape(1, nh * hd)
                aw_m = attn_w.softmax(-1)
                ao_m = torch.matmul(aw_m, v_t).transpose(0, 1).reshape(1, nh * hd)
                vo = d["attn_dump"][li]["attn_output"][s].to(dev).to(torch.float32)
                print(f"  [shape L3 out] ao_m={tuple(ao_m.shape)} ao_v={tuple(ao_v.shape)} "
                      f"vo={tuple(vo.shape)}")
                cm = F.cosine_similarity(ao_m.reshape(1,-1).float(), vo.reshape(1,-1), dim=-1).item()
                cv = F.cosine_similarity(ao_v.reshape(1,-1).float(), vo.reshape(1,-1), dim=-1).item()
                eqmax = (q_r - vq_r).abs().max().item()
                # per-head cos of vLLM-q attn output vs vLLM dump
                ao_v_h = ao_v.view(nh, hd); vo_h = vo.view(nh, hd)
                phc = F.cosine_similarity(ao_v_h, vo_h, dim=-1)
                print(f"  [decisive L3] myq_attn_cos={cm:.4f} vllmq_attn_cos={cv:.4f} "
                      f"q_r_vs_vq_maxdiff={eqmax:.5f}")
                print(f"  [perhead ao_v vs vo] cos={[round(x,2) for x in phc.tolist()]}")
                # also: attn weight argmax position per head (where does q attend?)
                awv = aw_v.squeeze(1)   # (nh, N)
                print(f"  [attn argmax pos] {awv.argmax(-1).tolist()}")
            # sliding-window mask: sliding layers only attend the last
            # `sliding_window` positions up to the query position s.
            sw = getattr(attn, "sliding_window", None) or args.sliding_window
            if lt == "sliding_attention" and sw:
                lo = max(0, s - int(sw) + 1)
                mask = torch.full((1, N), float("-inf"), device=attn_w.device,
                                  dtype=attn_w.dtype)
                mask[0, lo:s + 1] = 0.0
                attn_w = attn_w + mask
            attn_w = attn_w.softmax(dim=-1)
            attn_o = torch.matmul(attn_w, v_t)                 # (nh,1,hd)
            attn_o = attn_o.transpose(0, 1).reshape(1, nh * hd)
            # compare q(post-rope) and attn_output (pre o_proj) vs vLLM dump
            attn_dump = d.get("attn_dump")
            if attn_dump and li < len(attn_dump):
                ad = attn_dump[li]
                vq = ad["q_postrope"][s].to(dev).to(torch.float32)      # (nh*hd,)
                myq = q_r.transpose(0, 1).reshape(-1).to(torch.float32)
                cq = F.cosine_similarity(myq.unsqueeze(0), vq.unsqueeze(0), dim=-1).item()
                vo = ad["attn_output"][s].to(dev).to(torch.float32)
                myo = attn_o.reshape(-1).to(torch.float32)
                co = F.cosine_similarity(myo.unsqueeze(0), vo.unsqueeze(0), dim=-1).item()
                print(f"  [attn parity L{li}] q_cos={cq:.4f} attn_out_cos={co:.4f} "
                      f"my_kvheads={kvh} vllm_kvheads={ad['num_kv_heads']}")
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
            chk(f"layer{li}", hs[0])
            print(f"  [layer {li}] type={lt} hs_norm={hs.norm().item():.2f} "
                  f"attn_out_norm={attn_o.norm().item():.2f}")

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
