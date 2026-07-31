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

"""Mooncake store for Gemma4 MTP tensors.

Reuses ``EagleMooncakeStore``'s RDMA/host-buffer machinery (``_put_raw_tensors``,
``_get_tensors_gpu_direct`` / ``_get_tensors_via_host_buffer``) — those are
generic over a list of (key, tensor)/(name, shape, dtype) — and only overrides
the tensor manifest so we carry the Gemma4 MTP contract instead of Eagle3's
``_hs/_tgt/_ids/_lhs``:

    {key}_lh     last_hidden      (B, T, 2816)   bf16
    {key}_ids    input_ids        (B, T)         int64
    {key}_sk     sliding_k        (B, 8, T, 256) bf16
    {key}_sv     sliding_v        (B, 8, T, 256) bf16
    {key}_fk     full_k           (B, 2, T, 512) bf16
    {key}_fv     full_v           (B, 2, T, 512) bf16
    {key}_lm     loss_mask        (B, T)         (optional)

Shapes/dtypes are round-tripped via InferenceOutput.tensor_shapes/dtypes exactly
like Eagle3 so consumers can GET without prior knowledge of T.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from torchspec.transfer.mooncake.eagle_store import (
    HIDDEN_STATES_STORAGE_DTYPE,
    EagleMooncakeStore,
)
from torchspec.utils.logging import logger


class Gemma4MTPMooncakeStore(EagleMooncakeStore):
    """Mooncake store specialised for the Gemma4 MTP tensor set."""

    # (attribute_name_on_output, key_suffix, storage_dtype-or-None-to-keep)
    _TENSOR_LAYOUT: List[Tuple[str, str, Optional[torch.dtype]]] = [
        ("last_hidden", "_lh", HIDDEN_STATES_STORAGE_DTYPE),
        ("input_ids", "_ids", None),  # keep int64
        ("sliding_k", "_sk", HIDDEN_STATES_STORAGE_DTYPE),
        ("sliding_v", "_sv", HIDDEN_STATES_STORAGE_DTYPE),
        ("full_k", "_fk", HIDDEN_STATES_STORAGE_DTYPE),
        ("full_v", "_fv", HIDDEN_STATES_STORAGE_DTYPE),
    ]
    _OPTIONAL = [("loss_mask", "_lm", None)]

    def put(self, key: str, output) -> Dict[str, Any]:
        """Store a Gemma4MTPTargetOutput's tensors via async batch_put_from.

        Args:
            key: base mooncake key.
            output: Gemma4MTPTargetOutput (has the named tensor attributes).

        Returns:
            {"shapes": {...}, "dtypes": {...}} reflecting the stored tensors so
            the consumer can GET with correct shapes/dtypes.
        """
        self._ensure_initialized()

        keys: List[str] = []
        tensors: List[torch.Tensor] = []
        shapes: Dict[str, Tuple[int, ...]] = {}
        dtypes: Dict[str, torch.dtype] = {}

        layout = list(self._TENSOR_LAYOUT)
        if getattr(output, "loss_mask", None) is not None:
            layout = layout + self._OPTIONAL

        for attr, suffix, cast in layout:
            t = getattr(output, attr, None)
            if t is None:
                continue
            if cast is not None and t.dtype != cast:
                t = t.to(cast)
            keys.append(f"{key}{suffix}")
            tensors.append(t)
            shapes[attr] = tuple(t.shape)
            dtypes[attr] = t.dtype

        self._put_raw_tensors(keys, tensors)
        logger.debug("gemma4-mtp put: key=%s shapes=%s", key, shapes)
        return {"shapes": shapes, "dtypes": dtypes}

    def get(
        self,
        key: str,
        shapes: Dict[str, Tuple[int, ...]],
        dtypes: Dict[str, torch.dtype],
        device: torch.device,
    ):
        """Retrieve Gemma4 MTP tensors into GPU memory as a dict batch.

        Returns a plain dict (the trainer's _forward consumes named tensors),
        not an Eagle3TargetOutput. Keys: last_hidden, input_ids, sliding_k,
        sliding_v, full_k, full_v, and loss_mask if present.
        """
        self._ensure_initialized()

        layout = list(self._TENSOR_LAYOUT)
        if "loss_mask" in shapes:
            layout = layout + self._OPTIONAL

        keys: List[str] = []
        tensor_specs: List[Tuple[str, Tuple[int, ...], torch.dtype]] = []
        for attr, suffix, _cast in layout:
            if attr not in shapes:
                continue
            keys.append(f"{key}{suffix}")
            default_dtype = torch.int64 if attr == "input_ids" else HIDDEN_STATES_STORAGE_DTYPE
            tensor_specs.append((attr, shapes[attr], dtypes.get(attr, default_dtype)))

        tensor_map = None
        if self._gpu_direct_available and self._gpu_receive_buffer is not None:
            tensor_map = self._get_tensors_gpu_direct(keys, tensor_specs, device)
            if tensor_map is None:
                logger.warning("GPUDirect get failed; falling back to host buffer.")
        if tensor_map is None:
            tensor_map = self._get_tensors_via_host_buffer(keys, tensor_specs, device)

        logger.debug("gemma4-mtp get: key=%s -> %s", key, list(tensor_map.keys()))
        return tensor_map

    def remove_mtp_tensors(self, key: str, has_loss_mask: bool = False) -> None:
        """Force-delete all tensors for a Gemma4 MTP output (best-effort)."""
        suffixes = [s for _, s, _ in self._TENSOR_LAYOUT]
        if has_loss_mask:
            suffixes = suffixes + [s for _, s, _ in self._OPTIONAL]
        keys = [f"{key}{s}" for s in suffixes]
        try:
            self._store.batch_remove(keys, force=True)
        except Exception:
            logger.warning("gemma4-mtp force delete failed for %s", key, exc_info=True)
