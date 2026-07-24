#!/usr/bin/env bash
# Gemma4 MTP — two sequential experiments, run over the weekend.
#   EXP1: layer1_delta only  (single-layer smoke/sanity, ~13.5k train / 1k eval)
#   EXP2: full MaiProfile     (all 10 layers merged, ~325k train)
# Runs EXP1 first, then EXP2. Each saves to its OWN output_dir. Logs are tee'd
# so you can read either run's console after the fact.
#
# Usage:
#   cd /scratch/azureml/cr/j/<jobid>/exe/wd/TorchSpec
#   bash run_gemma4_mtp_experiments.sh            # both, in order
#   bash run_gemma4_mtp_experiments.sh layer1     # EXP1 only
#   bash run_gemma4_mtp_experiments.sh full       # EXP2 only
set -euo pipefail

# ----------------------------------------------------------------- environment
# ~/.local site-packages must win over the system cu13 build (see prior debugging).
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
# Socket NCCL on this node (no IB): keep P2P on NVLink, disable IB probing.
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1

# MaiProfile merged split mount (holds the full 10-layer train/eval).
MNT="${AZURE_ML_INPUT_ukwdata:-}/maiprofile"
FULL_TRAIN="$MNT/mtp_26b/split/train_maiprofile_26b.jsonl"
FULL_EVAL="$MNT/mtp_26b/split/eval_maiprofile_26b.jsonl"

# layer1_delta filtered files (produced by tools/gemma4_mtp/filter_layer.py).
# Absolute paths (resolved from this script's dir = TorchSpec root) so the
# existence check and the train_entry override use the SAME base — avoids the
# shell-CWD vs config-dir relative-path mismatch. Files live in ./data/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L1_TRAIN="$SCRIPT_DIR/data/train_layer1_delta.jsonl"
L1_EVAL="$SCRIPT_DIR/data/eval_layer1_delta_1000.jsonl"

CONFIG="configs/hf_gemma4_mtp.yaml"
# Shared overrides applied to BOTH experiments (kept here so experiment params
# are visible in one place, not buried in the yaml).
COMMON_OVERRIDES="training.max_seq_length=4096"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="./logs/gemma4_mtp"
mkdir -p "$LOGDIR"

# ----------------------------------------------------------------- helpers
run_exp () {
  local name="$1"; local train="$2"; local eval="$3"; local outdir="$4"; local extra="$5"
  local log="$LOGDIR/${name}_${TS}.log"
  echo "=================================================================="
  echo "[$(date '+%F %T')] START $name"
  echo "  train  = $train"
  echo "  eval   = $eval"
  echo "  output = $outdir"
  echo "  log    = $log"
  echo "=================================================================="
  if [[ ! -f "$train" ]]; then
    echo "!! train file missing: $train — SKIPPING $name" | tee -a "$log"
    return 1
  fi
  # shellcheck disable=SC2086
  python -m torchspec.train_entry --config "$CONFIG" \
      dataset.train_data_path="$train" \
      dataset.eval_data_path="$eval" \
      output_dir="$outdir" \
      cache_dir="./cache/${name}" \
      $extra \
      2>&1 | tee "$log"
  echo "[$(date '+%F %T')] DONE  $name -> model in $outdir"
}

# ----------------------------------------------------------------- experiments
MODE="${1:-both}"

if [[ "$MODE" == "layer1" || "$MODE" == "both" ]]; then
  # EXP1: single-layer. Eval subset is already 1000; keep the default cap.
  run_exp "exp1_layer1_delta" \
          "$L1_TRAIN" "$L1_EVAL" \
          "./outputs/gemma4-mtp/exp1_layer1_delta" \
          "$COMMON_OVERRIDES"
fi

if [[ "$MODE" == "full" || "$MODE" == "both" ]]; then
  # EXP2: full 10-layer merge. Full eval is huge (~32k); cap it to 1000 for a
  # comparable, bounded eval pass. Everything else identical to EXP1.
  run_exp "exp2_full" \
          "$FULL_TRAIN" "$FULL_EVAL" \
          "./outputs/gemma4-mtp/exp2_full" \
          "$COMMON_OVERRIDES dataset.max_eval_samples=1000"
fi

echo "[$(date '+%F %T')] ALL REQUESTED EXPERIMENTS FINISHED."
echo "Models:"
echo "  EXP1 layer1_delta : ./outputs/gemma4-mtp/exp1_layer1_delta"
echo "  EXP2 full         : ./outputs/gemma4-mtp/exp2_full"
