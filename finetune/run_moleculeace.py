"""Run SJoINT on MoleculeACE datasets.

Usage:
    python run_moleculeace.py --datasets CHEMBL204_Ki CHEMBL234_Ki
    python run_moleculeace.py --all
    python run_moleculeace.py --dry_run
"""
from __future__ import annotations
import argparse, os, sys, json, csv, subprocess, time
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_SCRIPT_DIR)
PYTHON = os.environ.get("SJOINT_PYTHON", "python")
TRAIN_SCRIPT = os.path.join(_SCRIPT_DIR, "train.py")

# Canonical pre-split CSVs (copied from MoleculeACE/Data/benchmark_data/old).
ACE_DATA = os.environ.get(
    "SJOINT_ACE_DATA",
    os.path.join(_REPO_DIR, "data", "MoleculeACE", "raw"),
)
# Pre-tensorised .pt outputs from `preprocess.moleculeace.tensorize`.
ACE_PROCESSED = os.environ.get(
    "SJOINT_ACE_PROCESSED",
    os.path.join(_REPO_DIR, "data", "MoleculeACE", "processed"),
)
RESULTS_BASE = os.path.join(_SCRIPT_DIR, "results", "moleculeace")

# Default to the general-purpose sweep best (override via env if needed).
PRETRAIN_CKPT = os.environ.get(
    "SJOINT_PRETRAIN_CKPT",
    os.path.join(_REPO_DIR, "pretrain", "sweep_results",
                 "lr0.0005_wd5e-05_bs512", "best_model.ckpt"),
)
VOCAB_PATH = os.environ.get(
    "SJOINT_VOCAB",
    os.path.join(_REPO_DIR, "data", "ZINC250K", "vocab.json"),
)

GPU_ID = int(os.environ.get("SJOINT_GPU", "0"))

# ESOL HP
ESOL_HP = {
    "stage1_epochs": 20,
    "learning_rate": 0.01,
    "weight_decay": 5e-5,
    "batch_size": 32,
    "num_head_layers": 2,
    "head_hidden": 128,
    "head_dropout": 0.2,
}

DATASETS_5 = [
    "CHEMBL204_Ki", "CHEMBL234_Ki", "CHEMBL2047_EC50",
    "CHEMBL2835_Ki", "CHEMBL239_EC50",
]

sys.path.insert(0, _REPO_DIR)
from preprocess.core.utils import load_vocab, process_smiles, to_sample


def prepare_dataset(dataset_name):
    """Convert MoleculeACE CSV to train.pt / test.pt for SJoINT."""
    csv_path = os.path.join(ACE_DATA, f"{dataset_name}.csv")
    out_dir = os.path.join(ACE_PROCESSED, dataset_name)

    train_pt = os.path.join(out_dir, "train.pt")
    test_pt = os.path.join(out_dir, "test.pt")

    if os.path.exists(train_pt) and os.path.exists(test_pt):
        print(f"  [SKIP] {dataset_name}: already preprocessed")
        return out_dir

    load_vocab(VOCAB_PATH)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    all_train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    # Split train into train/val (90/10)
    import random
    random.seed(42)
    shuffled = list(all_train_rows)
    random.shuffle(shuffled)
    val_size = max(1, len(shuffled) // 10)
    val_rows = shuffled[:val_size]
    train_rows = shuffled[val_size:]

    os.makedirs(out_dir, exist_ok=True)

    # Save metadata
    meta = {
        "dataset_name": dataset_name,
        "task_type": "regression",
        "label_cols": ["y"],
        "num_tasks": 1,
    }
    meta_dir = os.path.dirname(out_dir)
    with open(os.path.join(meta_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save cliff_mol info for evaluation
    cliff_info = {
        "test_cliff_mols": [int(r["cliff_mol"]) for r in test_rows],
        "test_smiles": [r["smiles"] for r in test_rows],
        "test_y": [float(r["y"]) for r in test_rows],
    }
    with open(os.path.join(out_dir, "cliff_info.json"), "w") as f:
        json.dump(cliff_info, f)

    for split_name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        samples, failed = [], 0
        for i, row in enumerate(split_rows):
            smi = row["smiles"].strip()
            label = [float(row["y"])]
            cpu_data = process_smiles(smi)
            if cpu_data is None:
                failed += 1
                continue
            samples.append(to_sample(cpu_data, label))
            if (i + 1) % 500 == 0:
                print(f"    {dataset_name}/{split_name}: {i+1}/{len(split_rows)}...")
        pt_path = os.path.join(out_dir, f"{split_name}.pt")
        torch.save(samples, pt_path, pickle_protocol=4)
        print(f"  {dataset_name}/{split_name}: {len(samples)} ok, {failed} failed")

    return out_dir


def train_and_eval(dataset_name, gpu_id):
    """Train SJoINT and return predictions."""
    data_dir = os.path.join(ACE_PROCESSED, dataset_name)
    hp = ESOL_HP
    variant = "moleculeace"

    cmd = [
        PYTHON, "-u", TRAIN_SCRIPT,
        "--dataset", dataset_name,
        "--seed", "1",
        "--gpu_id", str(gpu_id),
        "--num_workers", "4",
        "--learning_rate", str(hp["learning_rate"]),
        "--weight_decay", str(hp["weight_decay"]),
        "--batch_size", str(hp["batch_size"]),
        "--num_head_layers", str(hp["num_head_layers"]),
        "--head_hidden", str(hp["head_hidden"]),
        "--head_dropout", str(hp["head_dropout"]),
        "--stage1_epochs", str(hp["stage1_epochs"]),
        "--max_epochs", "200",
        "--early_stop_patience", "40",
        "--pretrain_ckpt", PRETRAIN_CKPT,
        "--variant", variant,
        "--data_base", ACE_PROCESSED,
    ]

    log_path = os.path.join(RESULTS_BASE, f"{dataset_name}.log")
    os.makedirs(RESULTS_BASE, exist_ok=True)

    print(f"  Training {dataset_name} on GPU {gpu_id}...")
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=_SCRIPT_DIR)

    if proc.returncode != 0:
        print(f"  FAIL: {dataset_name} (see {log_path})")
        return None

    # Read results
    results_dir = os.path.join(RESULTS_BASE, variant, dataset_name, "seed1")
    if not os.path.isdir(results_dir):
        print(f"  FAIL: results dir not found for {dataset_name}")
        return None

    # Find results.json
    for d in os.listdir(results_dir):
        rj = os.path.join(results_dir, d, "results.json")
        if os.path.exists(rj):
            with open(rj) as f:
                return json.load(f)
    return None


def calc_cliff_rmse(y_test, y_pred, cliff_mols):
    """Calculate RMSE on activity cliff molecules only."""
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    cliff_mols = np.array(cliff_mols, dtype=bool)
    if cliff_mols.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_test[cliff_mols] - y_pred[cliff_mols]) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS_5)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=GPU_ID)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--preprocess_only", action="store_true")
    args = parser.parse_args()

    if args.all:
        import glob
        args.datasets = [os.path.basename(f).replace(".csv", "")
                         for f in sorted(glob.glob(os.path.join(ACE_DATA, "*.csv")))]

    print("=" * 70)
    print(f"  MoleculeACE Evaluation — SJoINT")
    print(f"  Datasets: {len(args.datasets)}")
    print(f"  HP: ESOL (s1e{ESOL_HP['stage1_epochs']}_lr{ESOL_HP['learning_rate']})")
    print(f"  GPU: {args.gpu_id}")
    print("=" * 70)

    if args.dry_run:
        for ds in args.datasets:
            print(f"  {ds}")
        return

    # Step 1: Preprocess
    print("\n[Step 1] Preprocessing...")
    for ds in args.datasets:
        prepare_dataset(ds)

    if args.preprocess_only:
        return

    # Step 2: Train and evaluate
    print("\n[Step 2] Training...")
    results = {}
    for ds in args.datasets:
        r = train_and_eval(ds, args.gpu_id)
        if r:
            rmse = list(r["test"].values())[0]
            results[ds] = {"rmse": rmse}
            print(f"  {ds}: RMSE = {rmse:.4f}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  MoleculeACE Results — SJoINT")
    print(f"{'='*70}")
    print(f"  {'Dataset':<25s} {'RMSE':>8s}")
    print(f"  {'-'*25} {'-'*8}")
    for ds in args.datasets:
        if ds in results:
            print(f"  {ds:<25s} {results[ds]['rmse']:>8.4f}")
        else:
            print(f"  {ds:<25s}     FAIL")
    print("=" * 70)


if __name__ == "__main__":
    main()
