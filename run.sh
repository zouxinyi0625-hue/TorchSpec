#!/usr/bin/env bash
# Gemma4 MTP draft training — overnight run.
# Trains TWO drafts SEQUENTIALLY (both need all 8 GPUs, cannot overlap):
#   1) layer1_delta data
#   2) full maiprofile_26b data
# Each: warm-start from official assistant, strip forward, max_seq_length=4096.
# Checkpoints and logs go to SEPARATE paths so nothing collides.
#
# Usage:
#   cd $WD/TorchSpec && bash run.sh
# Run it detached so it survives logout:
#   cd $WD/TorchSpec && nohup bash run.sh > run_overnight.out 2>&1 &
#
set -uo pipefail

WD=/scratch/azureml/cr/j/27d05d9e68d844bd812d0ba6ed5c0577/exe/wd
cd "$WD/TorchSpec"
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

TARGET="$AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only"

# separate roots for checkpoints vs logs
CKPT_ROOT="$WD/TorchSpec/outputs"
LOG_ROOT="$WD/TorchSpec/train_logs"
mkdir -p "$LOG_ROOT"

STAMP=$(date +%Y%m%d_%H%M%S)

run_one () {
    local name="$1"           # run name
    local train="$2"          # train jsonl
    local eval="$3"           # eval jsonl
    local out="$CKPT_ROOT/$name"
    local cache="$WD/TorchSpec/cache/$name"
    local log="$LOG_ROOT/${name}_${STAMP}.log"

    echo "============================================================"
    echo "[$(date '+%F %T')] START $name"
    echo "  train = $train"
    echo "  eval  = $eval"
    echo "  ckpt  = $out"
    echo "  log   = $log"
    echo "============================================================"

    python -m torchspec.train_entry --config configs/hf_gemma4_mtp.yaml \
        model.target_model_path="$TARGET" \
        dataset.train_data_path="$train" \
        dataset.eval_data_path="$eval" \
        training.max_seq_length=4096 \
        cache_dir="$cache" \
        output_dir="$out" 2>&1 | tee "$log"

    local rc=${PIPESTATUS[0]}
    echo "[$(date '+%F %T')] END $name (exit=$rc)"
    echo "  checkpoints in: $out/checkpoints/"
    return $rc
}

# 1) layer1_delta
run_one "gemma4-mtp-4096-layer1" \
    "$WD/TorchSpec/data/train_layer1_delta.jsonl" \
    "$WD/TorchSpec/data/eval_layer1_delta_1000.jsonl"
echo "[$(date '+%F %T')] layer1 done, starting full data run..."

# 2) full maiprofile_26b
run_one "gemma4-mtp-4096-full" \
    "$AZURE_ML_INPUT_UKWDATA/maiprofile/mtp_26b/split/train_maiprofile_26b.jsonl" \
    "$AZURE_ML_INPUT_UKWDATA/maiprofile/mtp_26b/split/eval_maiprofile_26b.jsonl"

echo "============================================================"
echo "[$(date '+%F %T')] ALL RUNS DONE"
echo "  layer1 ckpt: $CKPT_ROOT/gemma4-mtp-4096-layer1/checkpoints/"
echo "  full   ckpt: $CKPT_ROOT/gemma4-mtp-4096-full/checkpoints/"
echo "  logs:        $LOG_ROOT/*_${STAMP}.log"
echo "  convert each with tools/gemma4_mtp/convert_dcp_to_hf.py before benching."
echo "============================================================"
