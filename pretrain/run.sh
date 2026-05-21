#!/bin/bash
# SJoINT pretrain — unified entry point.
#
# Usage:
#   bash pretrain/run.sh single        # Single training job (uses defaults below)
#   bash pretrain/run.sh sweep         # HP sweep across multiple GPUs (default 0,1)
#   bash pretrain/run.sh sweep --dry_run
set -e

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

DATA_PATH="${DATA_PATH:-$REPO_DIR/data/ZINC250K}"
# Make it absolute even if user passed relative
case "$DATA_PATH" in
  /*) ;;
  *) DATA_PATH="$REPO_DIR/$DATA_PATH" ;;
esac

case "${1:-}" in
  single)
    GPU_ID="${GPU_ID:-0}"
    LR="${LR:-5e-4}"
    WD="${WD:-5e-5}"
    BS="${BS:-512}"
    EPOCHS="${EPOCHS:-100}"
    OUT="${OUT:-$REPO_DIR/pretrain/checkpoints/lr${LR}_wd${WD}_bs${BS}}"
    case "$OUT" in
      /*) ;;
      *) OUT="$REPO_DIR/$OUT" ;;
    esac

    echo "=== Pretrain (single) ==="
    echo "  GPU       : $GPU_ID"
    echo "  Data      : $DATA_PATH"
    echo "  Output    : $OUT"
    echo "  lr/wd/bs  : $LR / $WD / $BS"
    echo "  epochs    : $EPOCHS"

    $PYTHON -u pretrain/train.py \
        --data_path "$DATA_PATH" \
        --checkpoint_dir "$OUT" \
        --gpu_id "$GPU_ID" \
        --learning_rate "$LR" \
        --weight_decay "$WD" \
        --batch_size "$BS" \
        --max_epochs "$EPOCHS"
    ;;

  sweep)
    shift  # remove "sweep"
    GPUS="${GPUS:-0,1}"
    PROCS_PER_GPU="${PROCS_PER_GPU:-10}"
    RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/pretrain/sweep_results}"
    # Make absolute even if user passed relative
    case "$RESULTS_DIR" in
      /*) ;;
      *) RESULTS_DIR="$REPO_DIR/$RESULTS_DIR" ;;
    esac

    echo "=== Pretrain HP sweep ==="
    echo "  GPUs           : $GPUS"
    echo "  Procs per GPU  : $PROCS_PER_GPU"
    echo "  Data           : $DATA_PATH"
    echo "  Results        : $RESULTS_DIR"

    SJOINT_PYTHON="$PYTHON" SJOINT_DATA="$DATA_PATH" \
        $PYTHON -u pretrain/run_pretrain_sweep.py \
            --gpus "$GPUS" \
            --procs_per_gpu "$PROCS_PER_GPU" \
            --results_dir "$RESULTS_DIR" \
            "$@"
    ;;

  *)
    echo "Usage: bash pretrain/run.sh {single|sweep} [extra args...]"
    echo ""
    echo "  single   Run one training job"
    echo "  sweep    HP grid sweep (lr × wd × bs = 4×3×2 = 24 combos)"
    echo ""
    echo "Environment variables:"
    echo "  PYTHON           Python binary             (default: sj env)"
    echo "  DATA_PATH        ZINC250K data dir         (default: data/ZINC250K)"
    echo "  GPU_ID           GPU id (single only)      (default: 0)"
    echo "  GPUS             GPU ids comma-sep (sweep) (default: 0,1)"
    echo "  PROCS_PER_GPU    Concurrent procs per GPU  (default: 10)"
    echo "  RESULTS_DIR      Sweep output dir          (default: pretrain/sweep_results)"
    echo "  LR/WD/BS/EPOCHS  Single-run hyperparameters"
    exit 1
    ;;
esac

echo
echo "Done."
