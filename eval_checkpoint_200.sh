#!/usr/bin/env bash
# Eval-only run: load a trained Gemma4 MTP checkpoint and evaluate it on a
# 200-sample subset of layer1_delta, WITHOUT retraining. Reports the training-
# side accept metric (eval/avg_acc, eval/simulated_acc_len) so you can compare
# the TRAINING-harness number against the vLLM DEPLOYMENT number on the SAME 200
# prompts.
#
# WHY 200: the full 1000-sample eval-cache gen is slow; 200 matches the online
# bench's --num-prompts 200 so the two numbers are on the same subset.
#
# WHAT IT DOES: runs the normal train_entry but with num_train_steps forced to 1
# and eval_interval=1, so the loop does one (throwaway) step then immediately
# evals the loaded checkpoint. The loaded weights are what get evaluated.
#
# RUN ON THE SERVER (8xA100), from the TorchSpec root.
#
# Usage:
#   bash eval_checkpoint_200.sh <checkpoint_parent_dir>
# where <checkpoint_parent_dir> is the dir CONTAINING iter_XXXXXXX (i.e. the
# .../checkpoints dir), e.g.
#   bash eval_checkpoint_200.sh \
#     $AZURE_ML_INPUT_UKWDATA/maiprofile/mtp_26b/gemma4_mtp_exp1_hf/checkpoints
# NOTE: load_path must point at the DCP checkpoints/ dir (the one holding
# iter_0002269 + latest_checkpointed_iteration.txt), NOT the hf_iter_* export.
set -euo pipefail

LOAD_PATH="${1:?usage: eval_checkpoint_200.sh <checkpoints_dir containing iter_XXXXXXX>}"

export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1

# Build the 200-sample eval subset (same source the online bench used).
SRC="data/eval_layer1_delta_1000.jsonl"
SUB="data/eval_layer1_delta_200.jsonl"
head -200 "$SRC" > "$SUB"
echo "eval subset: $(wc -l < "$SUB") samples -> $SUB"

CONFIG="configs/hf_gemma4_mtp.yaml"

python -m torchspec.train_entry --config "$CONFIG" \
    load_path="$LOAD_PATH" \
    dataset.eval_data_path="$SUB" \
    dataset.max_eval_samples=200 \
    dataset.eval_interval=1 \
    training.num_epochs=1 \
    training.max_seq_length=4096 \
    output_dir=./outputs/gemma4-mtp/eval_only \
    cache_dir=./cache/eval_only \
    2>&1 | tee logs/eval_checkpoint_200.log

echo ""
echo "Look for the line:  eval: loss=... acc=... sim_acc_len=..."
echo "and:                [MTP step 0] ... acc_per_step=[...]"
