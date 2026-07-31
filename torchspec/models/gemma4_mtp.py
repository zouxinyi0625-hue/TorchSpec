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

"""Gemma4 MTP training wrapper: multi-step drafting unroll + forward-KL loss.

This mirrors Eagle3Model's TTT loop (torchspec/models/eagle3.py) but follows the
EXACT Gemma4 assistant inference contract extracted from
``SinglePositionMultiTokenCandidateGenerator.get_candidates``
(docs/gemma4_mtp/design.md). The six invariants, and how this wrapper honours
each, per drafting step k = 0..K-1:

  1. q_len == 1                : each step drafts one token; we unroll K steps.
  2. position_ids constant     : ``position_ids`` is the last-seen position and
                                 never changes across the K steps.
  3. prev_hidden recurrence    : step 0 uses the TARGET's last hidden; step k>0
                                 uses the assistant's OWN post_projection output
                                 (``last_hidden_state``) fed back. ← the fix for
                                 "loss down / accept flat": the draft is trained
                                 on its own (imperfect) hidden, matching infer.
  4. target embedding table    : the token embedding is looked up in the TARGET
                                 model's embedding (raw/scaled), NOT the draft's.
  5. shared_kv_states fixed     : the same target KV dict is reused every step.
  6. cross-attn bidirectional  : handled inside the HF assistant forward.

Token feed on step k>0 (design decision D2): we TEACHER-FORCE the ground-truth
token at position t+k+1 (EAGLE-style). This gives clean, differentiable
supervision and is proven to raise accept; per-step accept is logged so we can
detect if later steps lag and switch to scheduled sampling.

Loss: forward-KL between draft logits and the TARGET distribution (accept is a
distribution-match criterion — NOT cross-entropy to ground truth). Per-step
accept broken out like Eagle3's ``acc_per_position``.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _forward_kl(draft_logits: torch.Tensor, target_p: torch.Tensor) -> torch.Tensor:
    """Per-token forward KL: E_target[log target - log draft], dropping the
    target-entropy term (constant wrt draft params) -> logsumexp(draft) -
    sum(target_p * draft_logits). Matches ops/loss._forward_kl_from_logits."""
    lg = draft_logits.float()
    return torch.logsumexp(lg, dim=-1) - (target_p * lg).sum(-1)


class Gemma4MTPModel(nn.Module):
    """Training wrapper around :class:`Gemma4MTPDraftModel`.

    Args:
        draft_model: the Gemma4MTPDraftModel (wraps HF assistant).
        mtp_num_steps: K, number of future tokens drafted per position.
        loss_decay_gamma: exp(-(k)/gamma) weighting so nearer steps dominate;
            set <=0 to disable (uniform weights).
        teacher_force: if True (default, D2), feed ground-truth token on steps
            k>0; if False, free-run the draft's own argmax token (matches infer
            exactly but non-differentiable at the token boundary).
    """

    def __init__(
        self,
        draft_model,
        mtp_num_steps: int = 4,
        loss_decay_gamma: float = 7.0,
        teacher_force: bool = True,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.K = int(mtp_num_steps)
        self.loss_decay_gamma = float(loss_decay_gamma)
        self.teacher_force = bool(teacher_force)
        self.backbone_hidden_size = draft_model.backbone_hidden_size

    def _step_target_p(
        self,
        target_last_hidden: torch.Tensor,  # (N, 2816) hidden at label positions
        target_lm_head_weight: torch.Tensor,  # (V, 2816)
    ) -> torch.Tensor:
        """Target next-token distribution at the label positions (detached)."""
        with torch.no_grad():
            tl = F.linear(target_last_hidden, target_lm_head_weight).float()
            return torch.softmax(tl, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,            # (B, T)
        target_last_hidden: torch.Tensor,   # (B, T, 2816) target hidden per position
        shared_kv_states: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        loss_mask: torch.Tensor,            # (B, T) 1 for supervised positions
        target_embed_weight: torch.Tensor,  # (V, 2816) TARGET embedding table (scaled)
        target_lm_head_weight: torch.Tensor,  # (V, 2816) TARGET lm_head (tied)
        attention_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the K-step MTP unroll and compute forward-KL loss.

        For every position t in the sequence, the draft predicts tokens
        t+1..t+K. We batch positions over the T axis: the assistant is called
        once per step on all positions in parallel, with the fixed shared_kv
        acting as cross-attention context (its bidirectional mask, built inside
        the HF forward, enforces that query position t attends to target KV
        prefix). position_ids per position is t itself (its "last seen" index),
        constant across the K steps (invariant 2).

        Returns:
            loss: scalar (decay-weighted mean forward-KL)
            accuracy: scalar top-1 match vs target argmax (binary, unweighted)
            loss_per_step: (K,) mean loss at each future step
            acc_per_step: (K,) mean accept (top-1 match) at each future step
            count_per_step: (K,) valid label count per step
        """
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        V = target_embed_weight.shape[0]

        # position_ids = each token's own index (its "last seen" position),
        # constant across the K drafting steps (invariant 2).
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)

        # step 0 prev_hidden = target's last hidden at each position (invariant 3).
        # NOTE: hidden STAYS hidden_t (unchanged) — only the token + label shift.
        prev_hidden = target_last_hidden
        # step 0 "last seen token" = token_{t+1} (the just-sampled next token),
        # NOT token_t. vLLM/HF deploy feed embed(token_{t+1}) paired with hidden_t
        # at anchor t (double iron-clad: vLLM proposer dump input_ids[t]==
        # target_token_ids[t+1], and HF 5.9 candidate_generator input_ids[:,-1:]).
        # Training previously fed token_t (off-by-one) → pos0 accept halved.
        cur_token_pos = (torch.arange(seqlen, device=device) + 1).clamp(max=seqlen - 1)
        cur_token = input_ids[:, cur_token_pos].clamp(0, V - 1)

        loss_per_step: List[torch.Tensor] = []
        acc_per_step: List[torch.Tensor] = []
        count_per_step: List[torch.Tensor] = []
        total_loss = input_ids.new_zeros((), dtype=torch.float32)
        total_correct = input_ids.new_zeros((), dtype=torch.float32)
        total_count = input_ids.new_zeros((), dtype=torch.float32)

        # Gemma scales input token embeddings by sqrt(hidden) inside embed_tokens
        # (embed_scale, applied in bf16). The MTP token half uses the TARGET 2816d
        # embedding table, so the scale is sqrt(2816)~=53.07 — NOT the draft's own
        # 1024d embed_scale (32.0). HF's assisted-decoding path builds inputs_embeds
        # from target_model_input_embeddings (the scaled embed_tokens module), so we
        # must reproduce that scale here; F.embedding alone is raw (norm ~1.6 vs the
        # trained ~85), which collapses the draft to noise (loss~=log V, acc~=0).
        embed_scale = torch.tensor(
            self.backbone_hidden_size ** 0.5, dtype=target_embed_weight.dtype
        )
        for k in range(self.K):
            # --- build inputs_embeds = concat[target_embed(cur_token), prev_hidden] ---
            tok_emb = F.embedding(cur_token, target_embed_weight) * embed_scale  # (B,T,2816) scaled
            inputs_embeds = torch.cat([tok_emb, prev_hidden], dim=-1)  # (B,T,5632)

            # --- assistant forward (delegates to HF; parity-verified) ---
            logits, last_hidden = self.draft_model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                shared_kv_states=shared_kv_states,
                attention_mask=attention_mask,
            )  # logits (B,T,V), last_hidden (B,T,2816)

            # --- labels (aligned to the vLLM/deploy shift) ---
            # After the +1 token shift, anchor position t consumes (token_{t+1},
            # hidden_t) and PREDICTS token_{t+k+2} at drafting step k. The target
            # hidden that predicts token_{t+k+2} sits at position (t+k+1), so the
            # supervising hidden uses shift=k+1 and the label token id sits at
            # t+k+2 (both +1 vs the old off-by-one alignment).
            hidden_pos = torch.arange(seqlen, device=device) + (k + 1)
            token_pos = torch.arange(seqlen, device=device) + (k + 2)
            valid_bounds = token_pos < seqlen           # label token must exist
            safe_hidden_pos = hidden_pos.clamp(max=seqlen - 1)
            safe_token_pos = token_pos.clamp(max=seqlen - 1)

            label_token = input_ids[:, safe_token_pos]          # (B,T) ground-truth next
            label_hidden = target_last_hidden[:, safe_hidden_pos, :]  # (B,T,2816)
            label_lossmask = loss_mask[:, safe_token_pos]       # (B,T)

            step_mask = (
                valid_bounds.unsqueeze(0).float()
                * label_lossmask.float()
            )  # (B,T)

            flat_mask = step_mask.reshape(-1)
            valid_idx = flat_mask.nonzero(as_tuple=True)[0]

            if valid_idx.numel() == 0:
                # Keep graph connected for FSDP even with no valid tokens.
                zero = logits.sum() * 0.0
                loss_per_step.append(zero.detach())
                acc_per_step.append(zero.detach())
                count_per_step.append(zero.detach())
            else:
                flat_logits = logits.reshape(-1, V).index_select(0, valid_idx)
                flat_label_hidden = label_hidden.reshape(-1, self.backbone_hidden_size).index_select(0, valid_idx)
                target_p = self._step_target_p(flat_label_hidden, target_lm_head_weight)

                tok_loss = _forward_kl(flat_logits, target_p)  # (Nvalid,)
                step_loss_sum = tok_loss.sum()
                step_count = torch.tensor(
                    float(valid_idx.numel()), device=device, dtype=torch.float32
                )

                with torch.no_grad():
                    draft_pred = flat_logits.argmax(-1)
                    tgt_pred = target_p.argmax(-1)
                    step_correct = (draft_pred == tgt_pred).float().sum()

                # decay weight: nearer steps weighted higher (k=0 -> 1.0).
                if self.loss_decay_gamma > 0:
                    w = float(torch.exp(torch.tensor(-k / self.loss_decay_gamma)))
                else:
                    w = 1.0
                total_loss = total_loss + w * step_loss_sum
                total_count = total_count + w * step_count
                total_correct = total_correct + step_correct

                loss_per_step.append((step_loss_sum / step_count).detach())
                acc_per_step.append((step_correct / step_count).detach())
                count_per_step.append(step_count.detach())

            # --- recurrence for next step ---
            prev_hidden = last_hidden  # own output fed back (invariant 3)
            if self.teacher_force:
                # feed the just-predicted ground-truth token (token_{t+k+2}, at
                # safe_token_pos after the +1 shift) — the teacher-forced analogue
                # of vLLM feeding the draft's own predicted next token at k>0.
                cur_token = input_ids[:, safe_token_pos].clamp(0, V - 1)
            else:
                # free-run: draft's own argmax (matches vLLM inference exactly,
                # which unconditionally feeds the draft's own prediction at k>0).
                cur_token = logits.argmax(-1).clamp(0, V - 1)

        loss = total_loss / total_count.clamp_min(1.0)
        accuracy = total_correct / torch.stack(
            [c for c in count_per_step]
        ).sum().clamp_min(1.0)

        return (
            loss,
            accuracy.detach(),
            torch.stack(loss_per_step),
            torch.stack(acc_per_step),
            torch.stack(count_per_step),
        )
