#!/bin/bash
# SJoINT preprocessing — unified entry point.
#
# Usage:
#   bash preprocess/run.sh pretrain      # Build JT vocab + tensorize ZINC250K
#   bash preprocess/run.sh moleculenet   # Split MoleculeNet CSVs + tensorize
#   bash preprocess/run.sh moleculeace   # Tensorize MoleculeACE (pre-split CSVs)
#
# `finetune` is kept as an alias for `moleculenet` for backward compatibility.
set -e

PYTHON="${PYTHON:-python}"
cd "$(dirname "$0")/.."

case "${1:-}" in
  pretrain)
    INPUT_TXT="${INPUT_TXT:-data/raw/zinc250k.txt}"
    OUT_PATH="${OUT_PATH:-data/processed/zinc250k/data}"
    VOCAB_PATH="${VOCAB_PATH:-data/vocab/vocab_zinc.json}"
    WORKERS="${WORKERS:-60}"
    CHUNK_SIZE="${CHUNK_SIZE:-50000}"

    echo "=== Step 1: Build JT vocab ==="
    if [ ! -f "$VOCAB_PATH" ]; then
        $PYTHON -m preprocess.pretrain.build_vocab \
            --input "$INPUT_TXT" \
            --output "$VOCAB_PATH" \
            --workers "$WORKERS"
    else
        echo "Vocab exists: $VOCAB_PATH"
    fi

    echo
    echo "=== Step 2: Tensorise ZINC SMILES → .pt chunks ==="
    $PYTHON -m preprocess.pretrain.tensorize \
        --input "$INPUT_TXT" \
        --output "$OUT_PATH" \
        --vocab "$VOCAB_PATH" \
        --workers "$WORKERS" \
        --chunk_size "$CHUNK_SIZE"
    ;;

  moleculenet|finetune)
    MODE="${MODE:-random}"
    CSV_DIR="${CSV_DIR:-data/raw/moleculenet}"
    VOCAB_PATH="${VOCAB_PATH:-data/vocab/vocab_zinc.json}"
    SEEDS="${SEEDS:-1 2 3 42 43 44 123 456 789 1024}"

    if [ "$MODE" = "both" ]; then
        OUT_DIR="${OUT_DIR:-data/processed/moleculenet/split}"
    elif [ "$MODE" = "scaffold" ]; then
        OUT_DIR="${OUT_DIR:-data/processed/moleculenet/scaffold_split}"
    else
        OUT_DIR="${OUT_DIR:-data/processed/moleculenet/random_split}"
    fi

    echo "=== Step 1: Split MoleculeNet CSVs (${MODE}) ==="
    echo "  Seeds: ${SEEDS}"
    $PYTHON -m preprocess.moleculenet.split \
        --mode "$MODE" \
        --csv_dir "$CSV_DIR" \
        --out_dir "$OUT_DIR" \
        --seeds $SEEDS

    echo
    echo "=== Step 2: Tensorise CSV → .pt ==="
    if [ "$MODE" = "both" ]; then
        $PYTHON -m preprocess.moleculenet.tensorize \
            --data_dir "$OUT_DIR" \
            --vocab "$VOCAB_PATH" \
            --modes random scaffold \
            --seeds $SEEDS
    else
        $PYTHON -m preprocess.moleculenet.tensorize \
            --data_dir "$OUT_DIR" \
            --vocab "$VOCAB_PATH" \
            --seeds $SEEDS
    fi
    ;;

  moleculeace)
    DATA_DIR="${DATA_DIR:-data/raw/moleculeace}"
    OUT_DIR="${OUT_DIR:-data/processed/moleculeace}"
    VOCAB_PATH="${VOCAB_PATH:-data/vocab/vocab_zinc.json}"
    WORKERS="${WORKERS:-30}"

    echo "=== Tensorise MoleculeACE (pre-split CSVs → .pt + cliff_mask) ==="
    $PYTHON -m preprocess.moleculeace.tensorize \
        --data_dir "$DATA_DIR" \
        --out_dir "$OUT_DIR" \
        --vocab "$VOCAB_PATH" \
        --workers "$WORKERS"
    ;;

  *)
    echo "Usage: bash preprocess/run.sh {pretrain|moleculenet|moleculeace}"
    echo ""
    echo "  pretrain     Build JT vocab + tensorize ZINC250K"
    echo "  moleculenet  Split MoleculeNet CSVs + tensorize  (alias: finetune)"
    echo "  moleculeace  Tensorize MoleculeACE (pre-split activity-cliff CSVs)"
    echo ""
    echo "Environment variables:"
    echo "  PYTHON     Python binary       (default: python)"
    echo "  MODE       Split mode          (moleculenet only; default: random; options: random / scaffold / both)"
    echo "  CSV_DIR    Raw CSV directory   (default: data/raw/moleculenet)"
    echo "  OUT_DIR    Output directory    (default depends on MODE)"
    echo "  SEEDS      Space-separated list of seeds (moleculenet only)"
    echo "             (default: 1 2 3 42 43 44 123 456 789 1024)"
    echo "  WORKERS    Parallel workers    (default: 60)"
    exit 1
    ;;
esac

echo
echo "Done."
