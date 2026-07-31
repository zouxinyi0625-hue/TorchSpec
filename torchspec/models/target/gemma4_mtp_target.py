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

"""Gemma4 MTP target model backend.

Unlike Eagle3 (which captures aux hidden states from several intermediate
layers), the Gemma4 assistant only needs, per training example:

  * ``last_hidden``: the target model's final hidden state at every position
    (B, T, 2816). Used both as step-0 prev_hidden and to compute the target
    next-token distribution (via the tied lm_head) for the forward-KL loss.
  * ``shared_kv_states``: the target model's KV of the LAST layer of each
    ``layer_type`` (full_attention / sliding_attention), which the assistant
    cross-attends to. Shapes (confirmed from the real target):
        sliding_attention: (B, 8, T, 256)  K and V
        full_attention:    (B, 2, T, 512)  K and V

The Gemma4 target exposes ``return_shared_kv_states=True`` on ``model(...)``
which returns exactly this dict, so we do not need forward hooks for the KV —
only a single forward. ``input_ids`` and ``loss_mask`` round-trip for the
trainer.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from torchspec.utils.distributed import get_tp_device_mesh, get_tp_group


@dataclass
class Gemma4MTPTargetOutput:
    """Tensors produced by the target for one Gemma4 MTP training example.

    ``shared_kv`` is flattened into individual tensors (rather than a nested
    dict) so it round-trips cleanly through the Mooncake key/tensor schema. The
    trainer/store reassembles the ``{layer_type: (K, V)}`` dict the assistant
    expects. Order is fixed: (sliding_K, sliding_V, full_K, full_V).
    """

    last_hidden: torch.Tensor          # (B, T, 2816)
    input_ids: torch.Tensor            # (B, T)
    sliding_k: torch.Tensor            # (B, 8, T, 256)
    sliding_v: torch.Tensor            # (B, 8, T, 256)
    full_k: torch.Tensor               # (B, 2, T, 512)
    full_v: torch.Tensor               # (B, 2, T, 512)
    loss_mask: Optional[torch.Tensor] = None  # (B, T)

    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        tensors = {
            "last_hidden": self.last_hidden,
            "input_ids": self.input_ids,
            "sliding_k": self.sliding_k,
            "sliding_v": self.sliding_v,
            "full_k": self.full_k,
            "full_v": self.full_v,
        }
        if self.loss_mask is not None:
            tensors["loss_mask"] = self.loss_mask
        return tensors

    def to_shared_kv(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Reassemble the {layer_type: (K, V)} dict the assistant consumes."""
        return {
            "sliding_attention": (self.sliding_k, self.sliding_v),
            "full_attention": (self.full_k, self.full_v),
        }


class Gemma4MTPTargetModel:
    """HuggingFace Gemma4 target backend for MTP data generation.

    Mirrors ``HFTargetModel`` (torchspec/models/target/eagle3_target_model.py)
    but produces the shared_kv + last_hidden contract instead of aux hidden
    states. A single forward with ``return_shared_kv_states=True`` yields
    everything; no forward hooks are needed.
    """

    #: layer_types whose LAST layer KV the assistant cross-attends to.
    SHARED_KV_LAYER_TYPES = ("sliding_attention", "full_attention")

    def __init__(self, model: nn.Module):
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "Gemma4MTPTargetModel":
        tp_group = get_tp_group()
        tp_size = tp_group.size() if tp_group is not None else 1

        if tp_size > 1:
            device_kwargs = {
                "tp_plan": "auto",
                "tp_size": tp_size,
                "device_mesh": get_tp_device_mesh(),
            }
        else:
            device_kwargs = {"device_map": device or "auto"}

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
            **device_kwargs,
            **kwargs,
        )
        return cls(model)

    def _inner_model(self) -> nn.Module:
        """Return the module exposing forward(..., return_shared_kv_states=True).

        For Gemma4ForCausalLM that is ``self.model.model`` (the Gemma4TextModel /
        multimodal wrapper). We probe common nestings defensively.
        """
        m = self.model
        for attr in ("model", "language_model"):
            inner = getattr(m, attr, None)
            if inner is not None and hasattr(inner, "forward"):
                m = inner
        return m

    @torch.no_grad()
    def generate_mtp_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Gemma4MTPTargetOutput:
        """Run the target once, returning last_hidden + shared_kv + ids.

        Args:
            input_ids: (B, T)
            attention_mask: (B, T) — forwarded to the target
            loss_mask: (B, T) — 1 for supervised positions, round-tripped

        Returns:
            Gemma4MTPTargetOutput with last_hidden (B,T,2816) and the four KV
            tensors of the last layer per layer_type.
        """
        inner = self._inner_model()
        out = inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_shared_kv_states=True,
        )

        last_hidden = out.last_hidden_state
        shared_kv = out.shared_kv_states
        if shared_kv is None:
            raise RuntimeError(
                "Target did not return shared_kv_states. Ensure the model is a "
                "Gemma4 target that supports return_shared_kv_states=True."
            )

        missing = [lt for lt in self.SHARED_KV_LAYER_TYPES if lt not in shared_kv]
        if missing:
            raise RuntimeError(
                f"shared_kv_states missing layer_types {missing}; "
                f"got {list(shared_kv.keys())}"
            )

        sliding_k, sliding_v = shared_kv["sliding_attention"]
        full_k, full_v = shared_kv["full_attention"]

        device = input_ids.device
        return Gemma4MTPTargetOutput(
            last_hidden=last_hidden.to(device),
            input_ids=input_ids,
            sliding_k=sliding_k.to(device),
            sliding_v=sliding_v.to(device),
            full_k=full_k.to(device),
            full_v=full_v.to(device),
            loss_mask=loss_mask.to(device) if loss_mask is not None else None,
        )
