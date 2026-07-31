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

"""Collator for Gemma4 MTP training batches.

Unlike the Eagle3 collator (which pads 2D/3D tensors with the sequence on dim
1), Gemma4 MTP carries the target shared-KV tensors of shape
``(B, num_kv_heads, T, head_dim)`` — the sequence is on dim 2. This collator
pads each tensor along its correct sequence axis and stacks the batch.

Tensor keys per sample (from Gemma4MTPMooncakeStore.get / dataset item):
    input_ids     (1, T)                 pad dim1
    loss_mask     (1, T)                 pad dim1
    last_hidden   (1, T, 2816)           pad dim1
    sliding_k/v   (1, 8, T, 256)         pad dim2
    full_k/v      (1, 2, T, 512)         pad dim2

All samples in a batch are padded to a common bucketed length to reduce unique
shapes (torch.compile / FlexAttention recompilation).
"""

from typing import Any, Dict, List

import torch

_BUCKET = 256


class Gemma4MTPCollator:
    """Pad-and-batch collator for Gemma4 MTP samples."""

    _SEQ_DIM1 = ("input_ids", "loss_mask")          # (B, T)
    _SEQ_DIM1_3D = ("last_hidden",)                  # (B, T, D)
    _SEQ_DIM2 = ("sliding_k", "sliding_v", "full_k", "full_v")  # (B, H, T, D)

    def __init__(self, usp_enabled: bool = False):
        # USP not supported for MTP yet; accept the arg for interface parity.
        if usp_enabled:
            raise NotImplementedError("USP is not supported for Gemma4 MTP training.")

    @staticmethod
    def _pad_dim1_2d(t: torch.Tensor, N: int) -> torch.Tensor:
        B, n = t.shape
        if n >= N:
            return t[:, :N]
        pad = torch.zeros(B, N - n, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=1)

    @staticmethod
    def _pad_dim1_3d(t: torch.Tensor, N: int) -> torch.Tensor:
        B, n, D = t.shape
        if n >= N:
            return t[:, :N, :]
        pad = torch.zeros(B, N - n, D, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=1)

    @staticmethod
    def _pad_dim2_4d(t: torch.Tensor, N: int) -> torch.Tensor:
        B, H, n, D = t.shape
        if n >= N:
            return t[:, :, :N, :]
        pad = torch.zeros(B, H, N - n, D, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=2)

    def _seq_len(self, item: Dict[str, Any]) -> int:
        ids = item["input_ids"]
        return ids.shape[1] if ids.dim() == 2 else ids.shape[0]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(self._seq_len(f) for f in features)
        max_len = ((max_len + _BUCKET - 1) // _BUCKET) * _BUCKET

        batch: Dict[str, Any] = {}

        for key in self._SEQ_DIM1:
            batch[key] = torch.cat(
                [self._pad_dim1_2d(f[key], max_len) for f in features], dim=0
            )
        for key in self._SEQ_DIM1_3D:
            batch[key] = torch.cat(
                [self._pad_dim1_3d(f[key], max_len) for f in features], dim=0
            )
        for key in self._SEQ_DIM2:
            batch[key] = torch.cat(
                [self._pad_dim2_4d(f[key], max_len) for f in features], dim=0
            )

        # attention_mask marks valid (non-pad) positions, dim1.
        batch["attention_mask"] = torch.cat(
            [
                torch.cat(
                    [
                        torch.ones(1, self._seq_len(f), dtype=torch.long),
                        torch.zeros(1, max_len - self._seq_len(f), dtype=torch.long),
                    ],
                    dim=1,
                )
                for f in features
            ],
            dim=0,
        ).to(batch["input_ids"].device)

        return batch
