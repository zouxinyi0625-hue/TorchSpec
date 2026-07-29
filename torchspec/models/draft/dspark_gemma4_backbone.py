# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Gemma4 DSpark draft backbone (training side).

Parallel to :mod:`torchspec.models.draft.dflash`'s Qwen3-style backbone, but with
Gemma4 attention operators. Same DFlash dual-source-KV block-diffusion mechanism
(Q from draft only; K/V = concat[context-from-target, draft]; block-causal
FlexAttention mask), swapped to Gemma4 semantics so that the trained draft matches
vLLM 0.26 ``gemma4_dspark.py::Gemma4DSparkForCausalLM`` at deployment.

Gemma4 specifics vs Qwen3 (ruler = vllm-026 gemma4_dspark.py + gemma4_mtp
strip-forward experience):
  - head_dim: full-attention layers use ``global_head_dim`` (independent of
    hidden/heads); dspark draft layers are ALL full_attention.
  - q_norm, k_norm, AND v_norm (v_norm has no learnable weight).
  - attention_k_eq_v: when set, V is derived from the K projection (v_proj=None,
    num_kv_heads=num_global_key_value_heads).
  - scaling = 1.0 (Gemma4 does not use 1/sqrt(head_dim) here).
  - scaled word embedding: embed_tokens(ids) * sqrt(hidden_size).
  - RoPE: Gemma4 full-attention rope over head_dim (partial_rotary handled by the
    rotary module dim; full layers here use the standard rope on global_head_dim).
  - context features: fc(concat[num_target_layers hidden]) -> hidden_norm.

This module intentionally reuses the FlexAttention block-mask + anchor/noise
machinery from the DFlash training wrapper unchanged; only the per-layer compute
is Gemma4-specific.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from torchspec.models.draft.dflash import (
    _repeat_kv,
    _rotate_half,
)


class Gemma4DSparkRMSNorm(nn.Module):
    """Gemma4 RMSNorm: (1 + weight) scaling, optional no-weight variant.

    Gemma normalizes then multiplies by ``(1 + weight)`` (weight initialized to 0),
    unlike the plain ``weight`` scaling in Qwen3/DFlashRMSNorm. ``has_weight=False``
    reproduces vLLM's ``RMSNorm(..., has_weight=False)`` used for gemma4 v_norm.
    """

    def __init__(self, dim: int, eps: float = 1e-6, has_weight: bool = True):
        super().__init__()
        self.eps = eps
        self.has_weight = has_weight
        if has_weight:
            self.weight = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.has_weight:
            x = x * (1.0 + self.weight.float())
        return x.to(dtype)


class Gemma4DSparkRotaryEmbedding(nn.Module):
    """Standard RoPE over ``head_dim`` for gemma4 full-attention draft layers."""

    def __init__(self, head_dim: int, max_position_embeddings: int = 32768,
                 base: float = 1_000_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings + 20
        self._build_cache(self.max_seq_len_cached)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_sin_cached", emb.sin(), persistent=False)

    def cos_sin_for(self, position_ids: torch.Tensor, device, dtype):
        """Return (cos, sin) gathered at ``position_ids`` → [B, 1, L, head_dim]."""
        cos = self._cos_cached.to(device)[position_ids]  # [B, L, head_dim]
        sin = self._sin_cached.to(device)[position_ids]
        return cos.unsqueeze(1).to(dtype), sin.unsqueeze(1).to(dtype)


class Gemma4DSparkAttention(nn.Module):
    """Dual-source KV attention with Gemma4 operators.

    Q from draft only; K/V = concat[context (target hidden), draft], projected by
    the SAME k/v projections. Gemma4: q_norm + k_norm + v_norm(no weight),
    global_head_dim for full-attention, optional attention_k_eq_v, scaling=1.0.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        # dspark draft layers are all full_attention → global_head_dim if present.
        self.head_dim = int(
            getattr(config, "global_head_dim", None)
            or getattr(config, "head_dim", self.hidden_size // self.num_heads)
        )
        self.use_k_eq_v = bool(getattr(config, "attention_k_eq_v", False))
        if self.use_k_eq_v:
            self.num_kv_heads = int(
                getattr(config, "num_global_key_value_heads", config.num_key_value_heads)
            )
        else:
            self.num_kv_heads = int(config.num_key_value_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = 1.0  # gemma4 dspark uses scaling=1.0
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        attn_bias = bool(getattr(config, "attention_bias", False))

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias)
        self.v_proj = (
            None
            if self.use_k_eq_v
            else nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias)
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=attn_bias)

        self.q_norm = Gemma4DSparkRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4DSparkRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4DSparkRMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)

        rope_theta = _resolve_rope_theta(config)
        self.rotary_emb = Gemma4DSparkRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=rope_theta,
        )

    def _project_kv(self, hidden: torch.Tensor):
        """Project hidden -> (k, v), each [B, L, num_kv_heads*head_dim] (pre-norm)."""
        k = self.k_proj(hidden)
        v = k if self.use_k_eq_v else self.v_proj(hidden)
        return k, v

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
    ) -> torch.Tensor:
        bsz, draft_len, _ = draft_hidden.shape
        ctx_len = context_hidden.shape[1]
        total_len = ctx_len + draft_len

        # Q from draft only
        q = self.q_proj(draft_hidden).view(bsz, draft_len, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [B, H, draft_len, hd]

        # K/V from context + draft (shared projections), concat before norm
        k_ctx, v_ctx = self._project_kv(context_hidden)
        k_draft, v_draft = self._project_kv(draft_hidden)
        k = torch.cat([k_ctx, k_draft], dim=1).view(bsz, total_len, self.num_kv_heads, self.head_dim)
        v = torch.cat([v_ctx, v_draft], dim=1).view(bsz, total_len, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)  # [B, KVH, total, hd]
        v = self.v_norm(v).transpose(1, 2)

        # RoPE: gather cos/sin at the respective positions
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        cos_q, sin_q = self.rotary_emb.cos_sin_for(draft_position_ids, q.device, q.dtype)
        cos_k, sin_k = self.rotary_emb.cos_sin_for(full_position_ids, k.device, k.dtype)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

        if block_mask is not None:
            from torchspec.models.ops.flex_attention import compile_friendly_flex_attention

            attn_output = compile_friendly_flex_attention(
                query=q,
                key=k,
                value=v,
                block_mask=block_mask,
                enable_gqa=True,
                scale=self.scaling,
            )
        else:
            k = _repeat_kv(k, self.num_kv_groups)
            v = _repeat_kv(v, self.num_kv_groups)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, dropout_p=0.0, scale=self.scaling
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, draft_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class Gemma4DSparkMLP(nn.Module):
    """Gemma4 MLP: gelu_pytorch_tanh gate activation."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        act = str(getattr(config, "hidden_activation", "gelu_pytorch_tanh"))
        self._use_gelu_tanh = "gelu" in act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_gelu_tanh:
            gate = F.gelu(self.gate_proj(x), approximate="tanh")
        else:
            gate = F.silu(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))


class Gemma4DSparkDecoderLayer(nn.Module):
    """Gemma4 decoder layer with dual-source KV attention (Gemma4 norm placement).

    Gemma4 uses pre+post norms around both attention and MLP sublayers
    (input_layernorm, post_attention_layernorm, pre_feedforward_layernorm,
    post_feedforward_layernorm). We mirror that so weights map 1:1 to the target
    assistant / vLLM draft.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.self_attn = Gemma4DSparkAttention(config)
        self.mlp = Gemma4DSparkMLP(config)
        eps = config.rms_norm_eps
        self.input_layernorm = Gemma4DSparkRMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = Gemma4DSparkRMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = Gemma4DSparkRMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = Gemma4DSparkRMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
    ) -> torch.Tensor:
        # attention sublayer: pre-norm in, post-norm out (gemma4 sandwich)
        residual = draft_hidden
        hidden = self.input_layernorm(draft_hidden)
        hidden = self.self_attn(
            draft_hidden=hidden,
            context_hidden=context_hidden,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
        )
        hidden = self.post_attention_layernorm(hidden)
        draft_hidden = residual + hidden

        # feedforward sublayer: pre-norm in, post-norm out
        residual = draft_hidden
        hidden = self.pre_feedforward_layernorm(draft_hidden)
        hidden = self.mlp(hidden)
        hidden = self.post_feedforward_layernorm(hidden)
        draft_hidden = residual + hidden
        return draft_hidden


def _resolve_rope_theta(config: PretrainedConfig) -> float:
    rope_params = getattr(config, "rope_parameters", None)
    if isinstance(rope_params, dict):
        for key in ("rope_theta", "theta", "base"):
            if key in rope_params:
                return float(rope_params[key])
    return float(getattr(config, "rope_theta", 1_000_000.0))


class Gemma4DSparkDraftModel(nn.Module):
    """Gemma4 DSpark draft backbone (Model level), interface-compatible with
    :class:`torchspec.models.draft.dspark.DSparkDraftModel`.

    Provides the exact contract the DFlash/DSpark training wrapper calls:
      - ``extract_context_feature(hidden_states_list)`` -> context_feature
      - ``forward(draft_input_ids, context_feature, draft_position_ids,
        context_position_ids, block_mask, noise_embedding)`` -> pre-norm-then-norm
        draft hidden states
      - ``embed_tokens`` / ``load_embedding`` / ``freeze_embedding``
      - ``markov_head`` / ``confidence_head`` (+ ``confidence_head_with_markov``)

    Gemma4 vs qwen3 DFlash backbone:
      - scaled word embedding: ``embed_tokens(ids) * sqrt(hidden_size)``.
      - context projection ``fc``: Linear(hidden*num_target_layers -> hidden),
        followed by ``hidden_norm`` (RMSNorm), matching vLLM ``Gemma4DSparkModel``.
      - Gemma4 decoder layers with dual-source KV.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.num_layers = int(config.num_hidden_layers)

        self.num_target_layers = int(getattr(config, "num_target_layers", 5))
        target_hidden_size = int(getattr(config, "target_hidden_size", self.hidden_size))
        self.target_hidden_size = target_hidden_size
        self.mask_token_id = int(getattr(config, "mask_token_id", config.vocab_size - 1))

        target_num_hidden = int(getattr(config, "target_num_hidden_layers", 36))
        self.target_layer_ids = getattr(config, "target_layer_ids", None)
        if self.target_layer_ids is None:
            self.target_layer_ids = build_target_layer_ids(
                self.num_target_layers, target_num_hidden
            )

        eps = float(config.rms_norm_eps)

        # scaled word embedding (gemma): embed_tokens(ids) * sqrt(hidden)
        self.embed_tokens = nn.Embedding(config.vocab_size, self.hidden_size)
        self.register_buffer(
            "embed_scale",
            torch.tensor(self.hidden_size**0.5, dtype=torch.float32),
            persistent=False,
        )

        # context feature projection: concat(num_target_layers hidden) -> hidden
        proj_input_dim = self.num_target_layers * target_hidden_size
        self.fc = nn.Linear(proj_input_dim, self.hidden_size, bias=False)
        self.hidden_norm = Gemma4DSparkRMSNorm(self.hidden_size, eps=eps)

        self.layers = nn.ModuleList(
            [Gemma4DSparkDecoderLayer(config) for _ in range(self.num_layers)]
        )
        self.norm = Gemma4DSparkRMSNorm(self.hidden_size, eps=eps)

        # ---- DSpark heads (reuse the architecture-agnostic implementations) ----
        from torchspec.models.draft.dspark import AcceptRatePredictor, build_markov_head

        self.markov_rank = int(getattr(config, "markov_rank", 0))
        self.confidence_head_with_markov = bool(
            getattr(config, "confidence_head_with_markov", True)
        )
        self.markov_head = build_markov_head(config)
        self.confidence_head: Optional[nn.Module] = None
        if getattr(config, "enable_confidence_head", False):
            conf_input_dim = self.hidden_size
            if self.confidence_head_with_markov:
                if self.markov_head is None:
                    raise ValueError(
                        "confidence_head_with_markov=True requires a Markov head "
                        "(markov_rank > 0)."
                    )
                conf_input_dim += self.markov_rank
            self.confidence_head = AcceptRatePredictor(conf_input_dim)

    # ------------------------------------------------------------------
    def extract_context_feature(self, all_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """concat(multi-layer target hidden) -> fc -> hidden_norm."""
        concatenated = torch.cat(all_hidden_states, dim=-1).to(self.fc.weight.dtype)
        return self.hidden_norm(self.fc(concatenated))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.embed_scale.to(self.embed_tokens.weight.dtype)

    def forward(
        self,
        draft_input_ids: Optional[torch.Tensor],
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        noise_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise_embedding is not None:
            draft_hidden = noise_embedding.to(context_feature.dtype)
        else:
            draft_hidden = self.embed_input_ids(draft_input_ids).to(context_feature.dtype)

        for layer in self.layers:
            draft_hidden = layer(
                draft_hidden=draft_hidden,
                context_hidden=context_feature,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
            )
        return self.norm(draft_hidden)

    # ------------------------------------------------------------------
    def freeze_embedding(self) -> None:
        self.embed_tokens.weight.requires_grad = False

    @torch.no_grad()
    def load_embedding(
        self, model_path: str, embedding_key: str = "model.embed_tokens.weight"
    ) -> None:
        """Load token embedding from the target checkpoint (reuses DFlash loader)."""
        from torchspec.models.draft.dflash import DFlashDraftModel

        DFlashDraftModel.load_embedding(self, model_path, embedding_key=embedding_key)
