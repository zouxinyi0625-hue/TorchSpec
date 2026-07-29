#!/usr/bin/env python
"""
Pure-PyTorch BATCH draft forward for Gemma4 MTP, ported from the validated
single-step strip (tools/gemma4_mtp/strip_vllm_draft_step0.py, which reproduces
vLLM's draft argmax bit-exactly). This is the training-side replacement for HF's
Gemma4AssistantForCausalLM parallel forward (which was proven NON-equivalent to
vLLM: test_hf_parallel_forward.py gave argmax 1201 != vLLM 188357).

KEY: every position t is an INDEPENDENT single query (matching vLLM's single-step
decode), vectorized over the whole sequence. Each query t attends the target's
shared_kv:
  - sliding layers: window [t-sliding_window+1, t]
  - full layer:     causal [0, t]
Per-layer-type partial rope via vLLM get_rope (full layer partial_rotary=0.25),
GQA against the cache's ACTUAL head count, scaling=1.0, embed x sqrt(2816).

This module has NO paged-attention / cudagraph / torch.compile — plain PyTorch,
so training and (strip-validated) deploy math are identical by construction.

VALIDATION (run this file directly on a strip dump, single sequence as batch=1):
  export PYTHONPATH=<vllm>:$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
  python torchspec/models/draft/gemma4_mtp_strip_forward.py \
      --official $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
      --target   $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
      --dump /tmp/vllm_draft_step0.pt
Expect: draft argmax @ sampled position == vLLM's dumped draft_token_ids (188357).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_module_with(root: nn.Module, attr: str):
    """BFS for the first submodule exposing ``attr`` (layout-agnostic)."""
    q = [root]
    while q:
        x = q.pop(0)
        if hasattr(x, attr):
            return getattr(x, attr)
        q.extend(list(x.children()))
    raise AttributeError(f"no submodule with attribute {attr!r}")


def _find_layers(root: nn.Module):
    """BFS for the decoder-layer ModuleList (the one holding self_attn layers)."""
    q = [root]
    while q:
        x = q.pop(0)
        if isinstance(x, nn.ModuleList) and len(x) > 0 and hasattr(x[0], "self_attn"):
            return x
        q.extend(list(x.children()))
    raise AttributeError("no decoder-layer ModuleList found")


def build_vllm_ropes(device, dtype=torch.bfloat16):
    """Build vLLM's own rope per layer-type (exact partial-rotary math).

    Must be built inside a vLLM config context (get_rope constructs a CustomOp).
    Returns {layer_type: RotaryEmbedding} moved to ``device``.
    """
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        ropes = {
            "sliding_attention": get_rope(
                head_size=256, max_position=131072, is_neox_style=True,
                dtype=dtype,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0}),
            "full_attention": get_rope(
                head_size=512, max_position=131072, is_neox_style=True,
                dtype=dtype,
                rope_parameters={"rope_type": "proportional",
                                 "rope_theta": 1000000.0,
                                 "partial_rotary_factor": 0.25}),
        }
    for r in ropes.values():
        r.to(device)
    return ropes


class Gemma4MTPStripForward(nn.Module):
    """Batch draft forward wrapping an HF Gemma4Assistant's SUBMODULES (weights)
    but running the strip's single-step-per-position attention (correct masks /
    partial rope / GQA), NOT HF's create_attention_masks parallel path.

    We reuse the HF module's parameters (pre_projection, per-layer submodules,
    norm, lm_head, post_projection) so weight loading stays identical; only the
    forward math is replaced.
    """

    def __init__(self, hf_assistant: nn.Module, sliding_window: int = 1024):
        super().__init__()
        self.m = hf_assistant  # Gemma4AssistantForCausalLM
        self.sliding_window = sliding_window
        self._ropes = None  # lazily built on first forward (needs device)

        # locate submodules by search (HF layout varies across versions), NOT by
        # hard-coded paths. Mirrors the validated strip script.
        self.pre_projection = _find_module_with(hf_assistant, "pre_projection")
        self.layers = _find_layers(hf_assistant)
        # parent module holding both .layers and .norm carries the final norm
        parent = None
        q = [hf_assistant]
        while q:
            x = q.pop(0)
            if hasattr(x, "layers") and hasattr(x, "norm"):
                parent = x
                break
            q.extend(list(x.children()))
        self.final_norm = parent.norm
        self.post_projection = _find_module_with(hf_assistant, "post_projection")
        self.lm_head = hf_assistant.get_output_embeddings()

    def _ropes_for(self, device, dtype):
        if self._ropes is None:
            self._ropes = build_vllm_ropes(device, dtype)
        return self._ropes

    def forward(
        self,
        inputs_embeds: torch.Tensor,     # (B, T, 2*backbone) = cat[tok_emb*scale, hidden]
        position_ids: torch.Tensor,      # (B, T)
        shared_kv_states: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        # each (K, V) as (B, kvh, T_kv, dim) in HF contract; we use per-position.
    ):
        """Return (logits (B,T,V), backbone_hidden (B,T,backbone))."""
        B, T, _ = inputs_embeds.shape
        device = inputs_embeds.device
        ropes = self._ropes_for(device, inputs_embeds.dtype)

        h = self._maybe_tuple(self.pre_projection(inputs_embeds))[0]  # (B,T,H)

        for layer in self.layers:
            attn = layer.self_attn
            lt = attn.layer_type
            hd = attn.head_dim
            nh = attn.config.num_attention_heads

            residual = h
            x = layer.input_layernorm(h)                      # (B,T,H)

            # q per position
            q = self._maybe_tuple(attn.q_proj(x))[0]          # (B,T,nh*hd)
            q = q.view(B, T, nh, hd)
            q = attn.q_norm(q)                                # (B,T,nh,hd)
            # rope: vLLM get_rope.forward_native expects (num_tokens, nh*hd)
            q_flat = q.reshape(B * T, nh * hd)
            pos_flat = position_ids.reshape(B * T)
            q_rot, _ = ropes[lt].forward_native(pos_flat, q_flat, None)
            q_rot = q_rot.view(B, T, nh, hd)                  # (B,T,nh,hd)

            # shared_kv: (B, kvh, T_kv, dim)
            k, v = shared_kv_states[lt]
            kvh = k.shape[1]
            T_kv = k.shape[2]

            # GQA: repeat kv heads to nh (against the cache's ACTUAL head count)
            rep = nh // kvh
            k_e = k.repeat_interleave(rep, dim=1)             # (B,nh,T_kv,dim)
            v_e = v.repeat_interleave(rep, dim=1)

            # scores: (B, nh, T_q, T_kv) — query t over kv positions
            q_bh = q_rot.permute(0, 2, 1, 3)                  # (B,nh,T,hd)
            scores = torch.matmul(q_bh, k_e.transpose(-1, -2))  # scaling=1.0

            # mask: query position pq attends kv position pk where
            #   full:    pk <= pq            (causal)
            #   sliding: pq-window+1 <= pk <= pq
            pq = position_ids.view(B, 1, T, 1)               # query positions
            pk = torch.arange(T_kv, device=device).view(1, 1, 1, T_kv)
            allowed = pk <= pq
            if lt == "sliding_attention":
                allowed = allowed & (pk > pq - self.sliding_window)
            scores = scores.masked_fill(~allowed, float("-inf"))

            probs = scores.softmax(dim=-1)
            attn_o = torch.matmul(probs, v_e)                # (B,nh,T,hd)
            attn_o = attn_o.permute(0, 2, 1, 3).reshape(B, T, nh * hd)

            attn_o = self._maybe_tuple(attn.o_proj(attn_o))[0]
            h = layer.post_attention_layernorm(attn_o) + residual

            residual = h
            x = layer.pre_feedforward_layernorm(h)
            x = self._maybe_tuple(layer.mlp(x))[0]
            x = layer.post_feedforward_layernorm(x)
            h = x + residual

            scalar = getattr(layer, "layer_scalar", None)
            if scalar is not None:
                h = h * scalar

        draft_hidden = self.final_norm(h)                    # (B,T,H)
        logits = self._maybe_tuple(self.lm_head(draft_hidden))[0]
        backbone_hidden = self._maybe_tuple(self.post_projection(draft_hidden))[0]
        return logits, backbone_hidden

    @staticmethod
    def _maybe_tuple(x):
        """HF layers sometimes return (tensor, ...) tuples; normalize to tensor."""
        if isinstance(x, tuple):
            return x
        return (x,)


# ---------------------------------------------------------------------------
# Standalone validation on a strip dump (batch=1 = the single dumped sequence).
# ---------------------------------------------------------------------------
def _find_embed(m):
    q = [m]
    while q:
        x = q.pop(0)
        if hasattr(x, "embed_tokens"):
            return x.embed_tokens.weight
        q.extend(list(x.children()))
    return None


def _main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--dump", default="/tmp/vllm_draft_step0.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--use-target-kv", action="store_true",
                    help="use target's dumped full-layer K/V (known-correct) "
                         "instead of Path-B gather for the full layer")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, Gemma4AssistantForCausalLM

    dev = args.device
    d = torch.load(args.dump, map_location="cpu")
    input_ids = d["input_ids"].to(dev)                       # (N,)
    hidden = d["hidden_states"].to(dev).to(torch.bfloat16)   # (N,2816)
    positions = d["positions"].to(dev).to(torch.long)        # (N,)
    tis = d["token_indices_to_sample"].flatten().tolist()
    vllm_draft = d["draft_token_ids"].flatten().tolist()
    N = input_ids.shape[0]
    s = tis[0]
    print(f"N={N} sampled={s} vllm_draft={vllm_draft}")

    # shared_kv in HF contract (B, heads, T_kv, dim). full <- tgt_fullkv (verified
    # correct); sliding <- gathered L28 (verified correct).
    g = d["shared_kv_gathered"]
    sl_key = [k for k in g if "28" in k][0]
    ks = g[sl_key]["k"].to(dev).to(torch.bfloat16)           # (N,8,256)
    vs = g[sl_key]["v"].to(dev).to(torch.bfloat16)
    ks = ks.permute(1, 0, 2).unsqueeze(0)                    # (1,8,N,256)
    vs = vs.permute(1, 0, 2).unsqueeze(0)

    tf = d["tgt_fullkv"][-1]
    nkvh_f, hd_f = tf["num_kv_heads"], tf["head_dim"]        # 2, 512
    kf = tf["k"].to(dev).to(torch.bfloat16).view(-1, nkvh_f, hd_f)
    vf = tf["v"].to(dev).to(torch.bfloat16).view(-1, nkvh_f, hd_f)
    kf = kf.permute(1, 0, 2).unsqueeze(0)                    # (1,2,N,512)
    vf = vf.permute(1, 0, 2).unsqueeze(0)

    shared_kv = {"sliding_attention": (ks, vs), "full_attention": (kf, vf)}
    print(f"shared_kv sliding={tuple(ks.shape)} full={tuple(kf.shape)}")

    # inputs_embeds = cat[ target_embed(input_ids)*sqrt(2816), hidden ]
    print("loading target embed ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, trust_remote_code=True).eval()
    embed_w = _find_embed(target).to(dev)
    scale = float(hidden.shape[-1] ** 0.5)
    tok_emb = F.embedding(input_ids, embed_w) * scale
    inputs_embeds = torch.cat([tok_emb, hidden], dim=-1).unsqueeze(0)  # (1,N,5632)
    del target

    print("loading draft ...")
    draft = Gemma4AssistantForCausalLM.from_pretrained(
        args.official, dtype=torch.bfloat16).to(dev).eval()
    fwd = Gemma4MTPStripForward(draft, sliding_window=1024).to(dev).eval()

    pos = positions.unsqueeze(0)
    with torch.no_grad():
        logits, _ = fwd(inputs_embeds, pos, shared_kv)
        argmax_s = int(logits[0, s].argmax().item())
        top5 = logits[0, s].topk(5).indices.tolist()

    print("=" * 60)
    print(f"strip-forward (batch) argmax @pos {s}: {argmax_s}")
    print(f"top5: {top5}")
    print(f"vLLM wants: {vllm_draft}")
    print("-" * 60)
    if argmax_s == vllm_draft[0]:
        print("=> MATCH: batch strip forward reproduces vLLM. Ready to wire into training.")
    else:
        print("=> DIFF: batch forward != vLLM. Compare against strip single-step.")
    print("=" * 60)


if __name__ == "__main__":
    _main()
