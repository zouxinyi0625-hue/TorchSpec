#!/usr/bin/env python
"""
Verify whether the HF Gemma4 target's shared_kv (return_shared_kv_states=True),
as produced by the ONLINE HF training pipeline (Gemma4MTPTargetModel), is
correct enough to reproduce vLLM's draft argmax when fed to the validated strip
forward.

Motivation: strip_vllm_draft_step0.py proved  vLLM-shared_kv + strip forward ->
188357. But the online HF config feeds HF-target shared_kv, NOT vLLM's. Since HF
target forward != vLLM target forward (hidden cos 0.97), the HF shared_kv may be
wrong. This settles it with data:

  run HF target on the SAME input_ids vLLM used (from the dump), take its
  last_hidden + shared_kv, feed the strip forward, compare argmax @ sampled pos
  against vLLM's dumped draft_token_ids (188357).

  MATCH  -> HF shared_kv is good enough; the online HF pipeline can train a
            train-inference-consistent draft (cheaper, no offline vLLM dump).
  DIFF   -> HF shared_kv differs from vLLM's; must offline-dump vLLM target
            data (hidden + shared_kv) for true train/infer parity.

Also prints cos(HF hidden, vLLM hidden) and cos(HF shared_kv, vLLM shared_kv)
per layer type so we see WHERE (hidden vs kv) HF diverges.

USAGE (server, GPU + transformers 5.9, vllm on PYTHONPATH for strip rope opt.):
  export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python tools/gemma4_mtp/verify_hf_shared_kv.py \
      --official $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
      --target   $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
      --dump /tmp/vllm_draft_step0.pt
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


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
    ap.add_argument("--dump", default="/tmp/vllm_draft_step0.pt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, Gemma4AssistantForCausalLM
    from torchspec.models.draft.gemma4_mtp_strip_forward import Gemma4MTPStripForward

    dev = args.device
    d = torch.load(args.dump, map_location="cpu")
    input_ids = d["input_ids"].to(dev)                       # (N,) vLLM's draft input
    positions = d["positions"].to(dev).to(torch.long)
    tis = d["token_indices_to_sample"].flatten().tolist()
    vllm_draft = d["draft_token_ids"].flatten().tolist()
    N = input_ids.shape[0]
    s = tis[0]
    print(f"N={N} sampled={s} vllm_draft={vllm_draft}")

    # ---- run HF target on the SAME input_ids to get HF hidden + HF shared_kv ----
    # NB: the dump's input_ids is the DRAFT input (shifted). For the target we
    # want the ORIGINAL token sequence. The dump also has target_token_ids —
    # that is what the target actually saw. Use it if present.
    tgt_ids = d.get("target_token_ids")
    if tgt_ids is not None:
        tgt_ids = tgt_ids.to(dev).long().view(1, -1)
    else:
        tgt_ids = input_ids.view(1, -1)
    print(f"target input_ids shape={tuple(tgt_ids.shape)}")

    print("loading HF target ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    embed_w = _find_embed(target).to(dev)

    # locate the inner model exposing return_shared_kv_states
    inner = target
    for attr in ("model", "language_model"):
        nxt = getattr(inner, attr, None)
        if nxt is not None and hasattr(nxt, "forward"):
            inner = nxt
    attn_mask = torch.ones_like(tgt_ids)
    with torch.no_grad():
        out = inner(input_ids=tgt_ids, attention_mask=attn_mask,
                    use_cache=True, return_shared_kv_states=True)
    hf_hidden = out.last_hidden_state[0].to(torch.bfloat16)  # (T,2816)
    hf_kv = out.shared_kv_states                              # {lt:(K,V)}
    print(f"HF hidden={tuple(hf_hidden.shape)} kv_keys={list(hf_kv.keys())}")
    for lt, (k, v) in hf_kv.items():
        print(f"  HF {lt}: K={tuple(k.shape)} V={tuple(v.shape)}")

    # ---- compare HF hidden / kv vs vLLM dumped ones ----
    vll_hidden = d["hidden_states"].to(dev).to(torch.float32)  # (N,2816)
    nmin = min(vll_hidden.shape[0], hf_hidden.shape[0])
    hc = F.cosine_similarity(hf_hidden[:nmin].float(), vll_hidden[:nmin], dim=-1)
    print(f"cos(HF hidden, vLLM hidden): mean={hc.mean():.4f} @s={hc[s]:.4f}")

    # vLLM shared_kv: sliding from gather (L28), full from tgt_fullkv
    g = d["shared_kv_gathered"]
    sl_key = [kk for kk in g if "28" in kk][0]
    vll_sl_k = g[sl_key]["k"].to(dev).float()               # (N,8,256)
    tf = d["tgt_fullkv"][-1]
    vll_fl_k = tf["k"].to(dev).float().view(-1, tf["num_kv_heads"], tf["head_dim"])  # (N,2,512)
    # HF kv are (B, heads, T, dim); take [0] and reorder to (T, heads, dim)
    hf_sl_k = hf_kv["sliding_attention"][0][0].permute(1, 0, 2).float()   # (T,8,256)
    hf_fl_k = hf_kv["full_attention"][0][0].permute(1, 0, 2).float()      # (T,2,512)
    for name, a, b in [("sliding K", hf_sl_k, vll_sl_k), ("full K", hf_fl_k, vll_fl_k)]:
        n = min(a.shape[0], b.shape[0])
        c = F.cosine_similarity(a[:n].reshape(n, -1), b[:n].reshape(n, -1), dim=-1)
        print(f"cos(HF {name}, vLLM {name}): mean={c.mean():.4f} @s={c[s]:.4f}")

    # ---- feed HF hidden + HF shared_kv to the strip forward ----
    scale = float(hf_hidden.shape[-1] ** 0.5)
    tok_emb = F.embedding(input_ids, embed_w) * scale
    inputs_embeds = torch.cat([tok_emb, hf_hidden], dim=-1).unsqueeze(0)  # (1,N,5632)

    # HF shared_kv already (B, heads, T, dim) — strip contract; slice to N
    shared_kv = {
        lt: (k[:, :, :N].to(torch.bfloat16), v[:, :, :N].to(torch.bfloat16))
        for lt, (k, v) in hf_kv.items()
    }

    del target
    print("loading draft ...")
    draft = Gemma4AssistantForCausalLM.from_pretrained(
        args.official, dtype=torch.bfloat16).to(dev).eval()
    fwd = Gemma4MTPStripForward(draft, sliding_window=1024, rope_impl="hf").to(dev).eval()

    with torch.no_grad():
        logits, _ = fwd(inputs_embeds, positions.unsqueeze(0), shared_kv)
        argmax_s = int(logits[0, s].argmax().item())
        top5 = logits[0, s].topk(5).indices.tolist()

    print("=" * 60)
    print(f"strip forward on HF shared_kv argmax @pos {s}: {argmax_s}")
    print(f"top5: {top5}")
    print(f"vLLM wants: {vllm_draft}")
    print("-" * 60)
    if argmax_s == vllm_draft[0]:
        print("=> MATCH: HF-target shared_kv reproduces vLLM. Online HF pipeline is")
        print("   train/infer consistent — can train without an offline vLLM dump.")
    else:
        print("=> DIFF: HF-target shared_kv does NOT reproduce vLLM. The online HF")
        print("   pipeline's data source diverges from deploy — must OFFLINE-dump")
        print("   vLLM target hidden+shared_kv for true parity. (See per-layer cos")
        print("   above for whether hidden or kv is the divergence.)")
    print("=" * 60)


if __name__ == "__main__":
    main()
