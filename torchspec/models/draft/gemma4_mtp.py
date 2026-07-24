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

"""Gemma4 MTP draft model: a thin TorchSpec wrapper around HF's
``Gemma4AssistantForCausalLM``.

Design rationale (see docs/gemma4_mtp/design.md):

We deliberately WRAP the HuggingFace assistant module rather than reimplement
its backbone, so the forward pass is bit-identical to Google's reference
(pre_projection → backbone(cross-attn over shared_kv_states) → post_projection
→ lm_head). This guarantees parity (verify_parity.py) and keeps us honest about
the six inference invariants extracted from
``SinglePositionMultiTokenCandidateGenerator.get_candidates``:

  1. q_len == 1 per assistant forward (autoregressive drafting)
  2. position_ids constant within a drafting round
  3. prev_hidden recurrence: step 0 = target last hidden; step t>0 = the
     assistant's OWN post_projection output (fed back)
  4. token embedding comes from the TARGET model's embedding table (raw/scaled)
  5. shared_kv_states fixed for the round (target KV, last layer per layer_type)
  6. cross-attention with bidirectional masks

The training wrapper (torchspec/models/gemma4_mtp.py) owns the multi-step unroll
and loss; this class only exposes a clean, parity-preserving forward plus the
helpers the trainer needs.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class Gemma4MTPConfig(PretrainedConfig):
    """Configuration for the Gemma4 MTP draft model.

    Mirrors the real assistant checkpoint at /tmp/models/gemma4/assistant:
      - draft backbone: hidden=1024, 4 layers, GQA 16/8 heads, head_dim=256
      - backbone(target) hidden = 2816
      - pre_projection: 2*2816=5632 → 1024   (concat[target_embed, prev_hidden])
      - post_projection: 1024 → 2816          (next-step prev_hidden)
      - lm_head: 1024 → 262144, tied embeddings

    The nested ``text_config`` (HF Gemma4AssistantConfig style) is preserved so
    we can hand it straight to the HF module. When absent we synthesise it from
    the flat fields below.
    """

    model_type = "gemma4_mtp"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        num_global_key_value_heads: int = 2,
        head_dim: int = 256,
        vocab_size: int = 262144,
        draft_vocab_size: int = 262144,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 262144,
        tie_word_embeddings: bool = True,
        backbone_hidden_size: int = 2816,
        mtp_num_steps: int = 4,
        use_shared_kv_states: bool = True,
        shared_kv_layer_types: Tuple[str, ...] = ("sliding_attention", "full_attention"),
        assistant_model_path: Optional[str] = None,
        target_model_path: Optional[str] = None,
        loss_objective: str = "kl",
        loss_decay_gamma: float = 7.0,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_global_key_value_heads = num_global_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.backbone_hidden_size = backbone_hidden_size
        self.mtp_num_steps = mtp_num_steps
        self.use_shared_kv_states = use_shared_kv_states
        self.shared_kv_layer_types = tuple(shared_kv_layer_types)
        self.assistant_model_path = assistant_model_path
        self.target_model_path = target_model_path
        self.loss_objective = loss_objective
        self.loss_decay_gamma = loss_decay_gamma
        # Convenience: pre/post projection dims implied by the design.
        self.pre_projection_in = 2 * backbone_hidden_size
        self.post_projection_out = backbone_hidden_size


class Gemma4MTPDraftModel(PreTrainedModel):
    """TorchSpec draft model wrapping HF ``Gemma4AssistantForCausalLM``.

    The wrapped HF module holds the real parameters (pre_projection,
    post_projection, 4-layer backbone, lm_head). We expose:

      * :meth:`forward` — one MTP step, bit-identical to the HF assistant.
      * :meth:`get_lm_head_params` — for the fused KL loss path (parity with
        the base draft interface).
      * :attr:`backbone_hidden_size` etc. — dims the trainer needs.

    Loading: the HF assistant weights are loaded from
    ``config.assistant_model_path`` (a local dir or hub id) when available;
    otherwise the module is initialised empty (init_empty_weights context) and
    weights are restored from a TorchSpec checkpoint by the trainer.
    """

    config_class = Gemma4MTPConfig
    base_model_prefix = "assistant"
    supports_gradient_checkpointing = True

    def __init__(self, config: Gemma4MTPConfig):
        super().__init__(config)
        self.backbone_hidden_size = config.backbone_hidden_size
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.target_vocab_size = config.vocab_size
        self.mtp_num_steps = config.mtp_num_steps
        self.shared_kv_layer_types = config.shared_kv_layer_types

        self.assistant = self._build_hf_assistant(config)

    # Convenience accessors — these are *properties*, not registered submodules,
    # so the underlying parameters live under a single name (self.assistant.*)
    # in state_dict. Registering them as module attributes (self.pre_projection
    # = self.assistant.pre_projection) creates aliases that break strict
    # load_state_dict (Missing key(s): draft_model.pre_projection.weight ...).
    @property
    def pre_projection(self) -> nn.Module:
        return self.assistant.pre_projection

    @property
    def post_projection(self) -> nn.Module:
        return self.assistant.post_projection

    @property
    def lm_head(self) -> nn.Module:
        return self.assistant.lm_head

    # ------------------------------------------------------------------ build
    @staticmethod
    def _build_hf_assistant(config: Gemma4MTPConfig) -> nn.Module:
        """Instantiate HF Gemma4AssistantForCausalLM from config.

        Most faithful path: when ``assistant_model_path`` is set, load the HF
        ``Gemma4AssistantConfig`` straight from the checkpoint so every field
        (hidden_size_per_layer_input, layer_types, rope_parameters, ...) matches
        exactly. Only synthesise a config as a fallback for the pathless case.
        Weights are loaded separately by ``load_assistant_weights``.
        """
        try:
            from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM
        except ImportError as e:  # pragma: no cover - depends on transformers>=5.9
            raise ImportError(
                "Gemma4 MTP requires transformers>=5.9 with Gemma4Assistant support. "
                f"Import failed: {e}"
            )

        path = getattr(config, "assistant_model_path", None)
        if path is not None:
            hf_cfg = Gemma4AssistantConfig.from_pretrained(path)
        else:
            hf_cfg = Gemma4MTPDraftModel._to_hf_assistant_config(config, Gemma4AssistantConfig)
        return Gemma4AssistantForCausalLM(hf_cfg)

    @staticmethod
    def _to_hf_assistant_config(config: Gemma4MTPConfig, hf_config_cls):
        """Map our flat/nested config onto an HF Gemma4AssistantConfig.

        If the original nested ``text_config`` was preserved on our config we
        pass it through untouched (most faithful). Otherwise we synthesise the
        text_config fields the assistant backbone needs from the flat fields.
        """
        text_config = getattr(config, "text_config", None)
        kwargs = {
            "backbone_hidden_size": config.backbone_hidden_size,
            "tie_word_embeddings": config.tie_word_embeddings,
        }
        if text_config is not None:
            kwargs["text_config"] = text_config
        else:
            kwargs["text_config"] = {
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "num_global_key_value_heads": config.num_global_key_value_heads,
                "head_dim": config.head_dim,
                "vocab_size": config.vocab_size,
                "rms_norm_eps": config.rms_norm_eps,
                "max_position_embeddings": config.max_position_embeddings,
                "tie_word_embeddings": config.tie_word_embeddings,
                "model_type": "gemma4_text",
                # Gemma4Assistant validator requires this to be 0 (no per-layer
                # input embeddings on the assistant backbone).
                "hidden_size_per_layer_input": 0,
                "enable_moe_block": False,
            }
        return hf_config_cls(**kwargs)

    @torch.no_grad()
    def load_assistant_weights(self, model_path: str) -> None:
        """Load HF assistant weights in-place from a local dir or hub id."""
        from transformers import Gemma4AssistantForCausalLM

        loaded = Gemma4AssistantForCausalLM.from_pretrained(
            model_path, dtype=self.assistant.dtype if hasattr(self.assistant, "dtype") else None
        )
        self.assistant.load_state_dict(loaded.state_dict(), strict=True)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        shared_kv_states: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One MTP drafting step — delegates to the HF assistant for parity.

        Args:
            inputs_embeds: (B, S, 2*backbone_hidden) = concat[target_embed(tok),
                prev_hidden]. NOT yet pre-projected (the HF module does that).
            position_ids: (B, S) — constant within a drafting round (invariant 2).
            shared_kv_states: {layer_type: (K, V)} target KV, last layer per type.
            attention_mask: optional base mask forwarded to create_attention_masks.

        Returns:
            logits: (B, S, vocab_size)
            last_hidden_state: (B, S, backbone_hidden) — post_projection output,
                to be fed back as prev_hidden on the next step (invariant 3).
        """
        out = self.assistant(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            shared_kv_states=shared_kv_states,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return out.logits, out.last_hidden_state

    # ------------------------------------------------------------- lm-head api
    def get_lm_head_params(self) -> Tuple[Optional[torch.Tensor], torch.Tensor, float]:
        """Return (norm_weight, lm_head_weight, norm_eps) for fused loss.

        The assistant applies its own final norm inside the backbone before
        producing last_hidden_state, and logits already come from lm_head. We
        expose the lm_head weight for any external logit recompute; norm is
        handled internally, so norm_weight is None and eps mirrors config.
        """
        return None, self.lm_head.weight, self.config.rms_norm_eps
