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

"""Gemma4 DSpark draft config derivation.

Derives a self-contained Gemma4 DSpark draft config from a Gemma4 (multimodal or
unified) target config. Mirrors DeepSpec's ``deepspec/modeling/dspark/gemma4/
config.py::build_draft_config`` so the produced draft matches vLLM 0.26's
``Gemma4DSparkForCausalLM`` deployment expectations.

Key gemma4 specifics (vs qwen3):
  - config lives under ``target_config.text_config`` (multimodal nesting).
  - draft layers are ALL ``full_attention`` (no sliding window in the draft).
  - carries gemma4 attention fields the backbone needs: global_head_dim,
    num_global_key_value_heads, attention_k_eq_v, head_dim, layer_types,
    rms_norm_eps, rope_parameters, hidden_activation.
  - draft is dense: ``enable_moe_block=False`` asserted (MoE target still gets a
    dense draft, per design).
"""

from __future__ import annotations

import copy
from typing import List, Optional


# gemma4 text fields the DSpark draft backbone / vLLM deployment rely on.
_REQUIRED_TEXT_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rms_norm_eps",
    "max_position_embeddings",
    "hidden_activation",
)

# gemma4-attention fields; some are absent on older/dense-only configs, so they
# are copied when present rather than hard-required.
_OPTIONAL_TEXT_FIELDS = (
    "global_head_dim",
    "num_global_key_value_heads",
    "attention_k_eq_v",
    "attention_bias",
    "attention_dropout",
    "enable_moe_block",
    "hidden_size_per_layer_input",
    "num_kv_shared_layers",
    "use_double_wide_mlp",
    "rope_parameters",
    "rope_theta",
    "layer_types",
    "sliding_window",
    "query_pre_attn_scalar",
    "attn_logit_softcapping",
    "final_logit_softcapping",
    "initializer_range",
)


def get_gemma4_text_config(target_config):
    """Return a deep copy of the gemma4 text sub-config from a target config.

    Accepts either a top-level gemma4/gemma4_unified config (with ``.text_config``)
    or a bare gemma4_text config (already the text level).
    """
    text_config = getattr(target_config, "text_config", None)
    if text_config is None:
        # already a text-level config
        text_config = target_config
    return copy.deepcopy(text_config)


def build_gemma4_dspark_draft_config_dict(
    target_config,
    *,
    num_draft_layers: int = 1,
    num_target_layers: Optional[int] = None,
    target_layer_ids: Optional[List[int]] = None,
    block_size: int = 7,
    num_anchors: int = 512,
    mask_token_id: int = 262144 - 1,
    markov_rank: int = 256,
    markov_head_type: str = "vanilla",
    enable_confidence_head: bool = True,
    confidence_head_with_markov: bool = True,
    draft_vocab_size: Optional[int] = None,
) -> dict:
    """Build a Gemma4 DSpark draft-config dict from a gemma4 target config.

    The returned dict is consumable by ``DSparkConfig(**dict)`` (it flows through
    ``PretrainedConfig`` kwargs). Field names/semantics mirror DeepSpec's
    ``build_draft_config`` and vLLM 0.26 ``gemma4_dspark.py`` expectations.
    """
    text = get_gemma4_text_config(target_config)

    for field in _REQUIRED_TEXT_FIELDS:
        assert hasattr(text, field), (
            f"target text_config.{field} must be provided for gemma4 dspark."
        )

    # draft is dense — a MoE target still gets a dense draft.
    if getattr(text, "enable_moe_block", False):
        # The draft backbone itself is dense; we simply never enable MoE on it.
        # (Do not assert-fail: the *target* may be MoE; we override on the draft.)
        pass

    target_num_layers = int(text.num_hidden_layers)
    if num_target_layers is None:
        num_target_layers = target_num_layers

    if target_layer_ids is None:
        target_layer_ids = _default_target_layer_ids(num_target_layers, target_num_layers)
    target_layer_ids = _validate_target_layer_ids(target_layer_ids, target_num_layers)

    # all draft layers are full_attention (no sliding window in the draft).
    layer_types = ["full_attention"] * int(num_draft_layers)

    cfg: dict = {}

    # copy required + optional gemma4 text fields
    for field in _REQUIRED_TEXT_FIELDS:
        cfg[field] = getattr(text, field)
    for field in _OPTIONAL_TEXT_FIELDS:
        if hasattr(text, field):
            cfg[field] = getattr(text, field)

    # draft-specific overrides
    cfg["architectures"] = ["Gemma4DSparkModel"]
    cfg["model_type"] = "gemma4_dspark"
    cfg["target_model_type"] = str(getattr(target_config, "model_type", "gemma4"))
    cfg["target_text_model_type"] = str(getattr(text, "model_type", "gemma4_text"))
    cfg["num_hidden_layers"] = int(num_draft_layers)
    cfg["num_target_layers"] = int(num_target_layers)
    cfg["target_hidden_size"] = int(text.hidden_size)
    cfg["target_num_hidden_layers"] = target_num_layers
    cfg["target_layer_ids"] = list(target_layer_ids)
    cfg["layer_types"] = layer_types
    cfg["enable_moe_block"] = False  # dense draft
    cfg["hidden_size_per_layer_input"] = int(
        getattr(text, "hidden_size_per_layer_input", 0)
    )
    cfg["tie_word_embeddings"] = False
    cfg["block_size"] = int(block_size)
    cfg["num_anchors"] = int(num_anchors)
    cfg["mask_token_id"] = int(mask_token_id)
    cfg["draft_vocab_size"] = int(draft_vocab_size or text.vocab_size)

    # DSpark head knobs
    cfg["markov_rank"] = int(markov_rank)
    cfg["markov_head_type"] = str(markov_head_type)
    cfg["enable_confidence_head"] = bool(enable_confidence_head)
    cfg["confidence_head_with_markov"] = bool(confidence_head_with_markov)

    return cfg


def _default_target_layer_ids(num_target_layers: int, num_hidden_layers: int) -> List[int]:
    """Evenly spaced target layer ids (SpecForge/DeepSpec convention).

    For num_target_layers=5, num_hidden_layers=L: pick low, then evenly spaced
    up to L-2 (avoid the very last layer, which is captured as last_hidden).
    """
    if num_target_layers <= 1:
        return [max(0, num_hidden_layers - 3)]
    if num_target_layers >= num_hidden_layers:
        return list(range(num_hidden_layers))
    # evenly spaced across [1, num_hidden_layers-2]
    lo, hi = 1, num_hidden_layers - 2
    step = (hi - lo) / (num_target_layers - 1)
    return [int(round(lo + i * step)) for i in range(num_target_layers)]


def _validate_target_layer_ids(layer_ids: List[int], num_hidden_layers: int) -> List[int]:
    ids = [int(x) for x in layer_ids]
    assert len(ids) > 0, "target_layer_ids must be non-empty."
    for x in ids:
        assert 0 <= x < num_hidden_layers, (
            f"target_layer_id {x} out of range [0, {num_hidden_layers})."
        )
    return ids


__all__ = [
    "build_gemma4_dspark_draft_config_dict",
    "get_gemma4_text_config",
]
