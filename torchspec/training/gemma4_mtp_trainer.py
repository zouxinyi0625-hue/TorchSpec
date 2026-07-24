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

"""Gemma4 MTP trainer — extends Trainer with MTP model init, forward, metrics.

Structurally mirrors DFlashTrainer (torchspec/training/dflash_trainer.py):
``init_model`` builds the draft (wrapping the HF assistant) under FSDP2, wraps it
in the training objective (Gemma4MTPModel), and sets up optimizer/scheduler.
``_forward`` reassembles shared_kv from the Mooncake tensors and runs the K-step
unroll. Metrics reuse the base per-step reduction helpers.

Key differences from DFlash:
  * The draft wraps HF ``Gemma4AssistantForCausalLM``; FSDP2 shards the inner
    backbone layers (``draft.assistant.model.layers``).
  * Both the TARGET embedding table (for input token embeds) and the TARGET
    lm_head (for the forward-KL target distribution) are needed; for Gemma4 they
    are tied, so one weight serves both.
  * The batch carries shared_kv tensors (sliding_k/v, full_k/v) + last_hidden
    instead of concatenated aux hidden states.
"""

from argparse import Namespace
from typing import Tuple

import torch
import torch.distributed as dist

from torchspec.models.draft.gemma4_mtp import Gemma4MTPConfig, Gemma4MTPDraftModel
from torchspec.models.gemma4_mtp import Gemma4MTPModel
from torchspec.training import checkpoint
from torchspec.training.dflash_trainer import DFlashTrainer
from torchspec.training.fsdp import apply_fsdp2, fsdp2_load_full_state_dict
from torchspec.training.optimizer import BF16Optimizer
from torchspec.utils.distributed import get_gloo_group
from torchspec.utils.logging import logger


class Gemma4MTPTrainer(DFlashTrainer):
    """Trainer for the Gemma4 MTP draft (assistant).

    Inherits DFlashTrainer's generic training loop, per-position metric
    reduction, ``_train_step`` and ``_aggregate_metrics`` (MTP's ``_forward``
    returns the same 6-tuple contract). Overrides only model/store/collator
    construction, ``init_model`` and ``_forward``.
    """

    _draft_config_class = Gemma4MTPConfig
    # MTP supervises a real predicted token at every unroll step — there is no
    # anchor slot to drop (unlike DFlash's index-0 anchor).
    _anchor_slot_offset = 0

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.mtp_num_steps = getattr(args, "gemma4_mtp_num_steps", 4)
        self.loss_decay_gamma = getattr(args, "gemma4_mtp_loss_decay_gamma", 7.0)
        self.teacher_force = getattr(args, "gemma4_mtp_teacher_force", True)
        # TARGET weights (tied embed == lm_head for Gemma4). Filled in init_model.
        self.target_embed_weight: torch.Tensor | None = None
        self.target_lm_head_weight: torch.Tensor | None = None

    # ------------------------------------------------------------------ build
    def _get_init_weight_context_manager(self):
        """All ranks build the draft on real CPU weights (no meta device).

        The base class puts non-rank-0 processes on meta device and relies on
        fsdp2_load_full_state_dict to materialize them from rank 0's broadcast.
        That works for the custom small drafts (DFlash/Eagle3) but NOT for our
        draft, which wraps HF ``Gemma4AssistantForCausalLM``: HF module init on
        meta device leaves some params unmaterialized after FSDP load, so the
        optimizer's clone() on ranks 1..N-1 blocks on never-completed
        materialization while rank 0 sails through — the exact rank-drift
        deadlock py-spy caught (ranks 2-6 stuck in optimizer.clone, ranks 0/1 at
        the finalize_load barrier). The draft is only 419M, so real CPU init on
        every rank is cheap and removes the meta-device hazard entirely.
        """

        def cpu_init_weights():
            return torch.device("cpu")

        return cpu_init_weights

    def _make_mooncake_store(self, mooncake_config):
        from torchspec.transfer.mooncake.gemma4_mtp_store import Gemma4MTPMooncakeStore

        return Gemma4MTPMooncakeStore(mooncake_config)

    def _make_collator(self, usp_enabled: bool):
        from torchspec.data.gemma4_mtp_collator import Gemma4MTPCollator

        return Gemma4MTPCollator(usp_enabled=usp_enabled)

    def _build_draft_model(self, config: Gemma4MTPConfig) -> Gemma4MTPDraftModel:
        return Gemma4MTPDraftModel(config)

    def _build_training_wrapper(self, draft_model: Gemma4MTPDraftModel) -> Gemma4MTPModel:
        return Gemma4MTPModel(
            draft_model=draft_model,
            mtp_num_steps=self.mtp_num_steps,
            loss_decay_gamma=self.loss_decay_gamma,
            teacher_force=self.teacher_force,
        )

    def init_model(
        self,
        draft_model_config,
        target_model_path: str,
        mooncake_config=None,
    ) -> int:
        if mooncake_config is not None:
            from torchspec.transfer.mooncake.utils import check_mooncake_master_available

            check_mooncake_master_available(
                mooncake_config.master_server_address, mooncake_config.metadata_server
            )

        init_context = self._get_init_weight_context_manager()

        with init_context():
            cfg_cls = self._draft_config_class
            if isinstance(draft_model_config, str):
                config = cfg_cls.from_pretrained(draft_model_config)
            elif isinstance(draft_model_config, dict):
                config = cfg_cls(**draft_model_config)
            elif isinstance(draft_model_config, Gemma4MTPConfig):
                config = draft_model_config
            else:
                raise TypeError(
                    f"Unsupported draft_model_config type: {type(draft_model_config).__name__}. "
                    f"Expected str, dict, or Gemma4MTPConfig."
                )

            # Ensure the target path is available so the wrapper can load the HF
            # assistant config/weights faithfully.
            if getattr(config, "target_model_path", None) is None:
                config.target_model_path = target_model_path

            draft_model = self._build_draft_model(config)

        # Load real assistant weights on rank 0 (warm start from the pretrained
        # assistant); FSDP broadcasts the full state dict below.
        assistant_path = getattr(config, "assistant_model_path", None)
        if assistant_path is not None and dist.get_rank() == 0:
            draft_model.load_assistant_weights(assistant_path)
            logger.info(f"[Rank 0] Loaded HF assistant weights from {assistant_path}")

        draft_model = draft_model.to(torch.bfloat16)
        dist.barrier(group=get_gloo_group())

        trainable = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        logger.info(f"[Rank {self.dp_rank}] Gemma4 MTP draft: {trainable:,} trainable params")

        mtp_model = self._build_training_wrapper(draft_model)

        full_state = mtp_model.state_dict() if dist.get_rank() == 0 else {}

        # FSDP2: shard the assistant backbone decoder layers.
        modules_to_shard = list(draft_model.assistant.model.layers)
        mtp_model = apply_fsdp2(
            mtp_model,
            mesh=self.dp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            modules_to_shard=modules_to_shard,
        )
        mtp_model = fsdp2_load_full_state_dict(
            mtp_model,
            full_state,
            self.dp_mesh,
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        # fsdp2_load_full_state_dict broadcasts params from rank0 with
        # non_blocking=True; the transfers are still in flight when the
        # optimizer clones the params for its fp32 master copy. On non-rank0
        # the clone reads not-yet-filled tensors and blocks on the pending
        # broadcast while rank0 races ahead to the next barrier -> cross-rank
        # deadlock (100% util, 120W spin). Force the broadcast to land and
        # align all ranks before optimizer construction.
        torch.cuda.synchronize()
        dist.barrier(group=get_gloo_group())

        if getattr(self.args, "compile_model", False):
            logger.info("Compiling Gemma4 MTP model with torch.compile")
            mtp_model = torch.compile(mtp_model)

        self.model = mtp_model
        _unwrapped = getattr(self.model, "_orig_mod", self.model)
        self.mtp = getattr(_unwrapped, "module", _unwrapped)
        self.draft_model = self.mtp.draft_model

        total_steps = self.args.lr_total_steps
        decay_style = getattr(self.args, "lr_decay_style", "cosine")
        warmup_ratio = getattr(self.args, "warmup_ratio", 0.1)
        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 0.0),
            max_grad_norm=self.args.max_grad_norm,
            warmup_ratio=warmup_ratio,
            total_steps=total_steps,
            decay_style=decay_style if decay_style != "WSD" else "cosine",
            min_lr=getattr(self.args, "min_lr", 0.0),
        )
        self.lr_scheduler = self.optimizer.lr_scheduler

        checkpoint_payload = checkpoint.load(self)
        checkpoint.finalize_load(self, checkpoint_payload)

        self._init_target_weights(target_model_path)

        self.prof.on_init_end()
        logger.info(f"[Rank {self.dp_rank}] Gemma4 MTP model initialized with FSDP2")
        return 0

    # ---------------------------------------------------------- target weights
    def _init_target_weights(self, target_model_path: str) -> None:
        """Load the TARGET embedding table (== tied lm_head) used for the input
        token embeddings AND the forward-KL target distribution.

        For Gemma4 tie_word_embeddings=True, so embed and lm_head share one
        weight. We load the embedding once on all ranks (it is frozen).
        """
        from torchspec.models.target.target_utils import TargetLMHead

        # lm_head weight (V, 2816); tied so it equals the embedding table.
        head = TargetLMHead.from_pretrained(
            model_path=target_model_path,
            lm_head_key=getattr(self.args, "lm_head_key", "lm_head.weight"),
            device="cuda",
            dtype=torch.bfloat16,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )
        w = head.lm_head.weight.detach()
        self.target_lm_head_weight = w
        self.target_embed_weight = w  # tied
        logger.info(
            f"[Rank {self.dp_rank}] Target embed/lm_head weight loaded {tuple(w.shape)}"
        )

    # ---------------------------------------------------------------- forward
    def _forward(
        self, batch: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        device = torch.device("cuda")
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        last_hidden = batch["last_hidden"].to(device, non_blocking=True)

        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        loss_mask = loss_mask.to(device, non_blocking=True)

        shared_kv = {
            "sliding_attention": (
                batch["sliding_k"].to(device, non_blocking=True),
                batch["sliding_v"].to(device, non_blocking=True),
            ),
            "full_attention": (
                batch["full_k"].to(device, non_blocking=True),
                batch["full_v"].to(device, non_blocking=True),
            ),
        }

        loss, accuracy, loss_per_step, acc_per_step, count_per_step = self.model(
            input_ids=input_ids,
            target_last_hidden=last_hidden,
            shared_kv_states=shared_kv,
            loss_mask=loss_mask,
            target_embed_weight=self.target_embed_weight,
            target_lm_head_weight=self.target_lm_head_weight,
        )
        return loss, accuracy, loss_per_step, acc_per_step, count_per_step, {}

    def _backward(self, loss: torch.Tensor, accumulation_steps: int = 1) -> torch.Tensor:
        (loss / accumulation_steps).backward()
        return loss

    def _aggregate_metrics(
        self, all_step_metrics: list[dict], step: int, *, grad_norm: torch.Tensor = None
    ) -> dict:
        # Reuse DFlash's aggregation (identical per-position contract), then log
        # the per-step accept/loss breakdown to the console on early steps so the
        # step-0 accept curve is visible without opening wandb.
        metrics = super()._aggregate_metrics(all_step_metrics, step, grad_norm=grad_norm)
        if dist.get_rank() == 0 and (step <= 5 or step % 50 == 0):
            n = self.mtp_num_steps
            accs = [metrics.get(f"train/acc_{i}") for i in range(n)]
            losses = [metrics.get(f"train/ploss_{i}") for i in range(n)]
            acc_str = ", ".join(f"{a:.3f}" if a is not None else "—" for a in accs)
            loss_str = ", ".join(f"{l:.3f}" if l is not None else "—" for l in losses)
            logger.info(
                f"[MTP step {step}] avg_loss={metrics.get('train/avg_loss', 0):.4f} "
                f"avg_acc={metrics.get('train/avg_acc', 0):.4f} "
                f"sim_acc_len={metrics.get('train/simulated_acc_len', 0):.2f} | "
                f"acc_per_step=[{acc_str}] loss_per_step=[{loss_str}]"
            )
        return metrics
