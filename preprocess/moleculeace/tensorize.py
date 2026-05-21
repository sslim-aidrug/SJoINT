"""MoleculeACE preprocessing — preserve the official train/test split.

Reads the canonical pre-split CSVs (with `split` column = train/test) shipped
with MoleculeACE [van Tilborg et al., JCIM 2022] and tensorises each fold to
.pt for SJoINT fine-tuning. A 10% val fold is carved off the canonical train
deterministically (first 10% by file order) — this matches the convention used
in `finetune/run_moleculeace.py`, so SJoINT's MoleculeACE numbers stay
comparable to prior runs.

Output layout (per ChEMBL task):
    out_dir/<TASK>/train.pt
    out_dir/<TASK>/val.pt
    out_dir/<TASK>/test.pt
    out_dir/<TASK>/cliff_metadata.json   # cliff_mols, smiles, y for the test set
    out_dir/metadata.json                 # task type / num_tasks (regression, 1)

Usage:
    python -m preprocess.moleculeace.tensorize \\
        --csv_dir data/MoleculeACE/raw \\
        --out_dir data/MoleculeACE/processed \\
        --vocab data/ZINC250K/vocab.json
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import torch

from preprocess.core.utils import load_vocab, process_smiles, to_sample

# Canonical val fraction inside the MoleculeACE train fold.
VAL_FRAC = 0.1


def discover_tasks(csv_dir):
    return sorted([f[:-4] for f in os.listdir(csv_dir) if f.endswith(".csv")])


def load_split_csv(csv_path):
    """Read MoleculeACE CSV and return (train_rows, val_rows, test_rows)."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows or "split" not in rows[0]:
        raise ValueError(f"{csv_path}: missing 'split' column — expected canonical pre-split CSV")

    all_train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]

    val_size = max(1, int(len(all_train) * VAL_FRAC))
    val = all_train[:val_size]
    train = all_train[val_size:]
    return train, val, test


def _worker(item):
    smi, y = item
    cpu = process_smiles(smi)
    if cpu is None:
        return None
    return to_sample(cpu, [y])


def tensorize_rows(rows, num_workers):
    items = [(r["smiles"], float(r["y"])) for r in rows]
    samples, failed = [], 0
    with mp.Pool(processes=num_workers) as pool:
        for s in pool.imap(_worker, items, chunksize=64):
            if s is None:
                failed += 1
            else:
                samples.append(s)
    return samples, failed


def process_task(task, csv_path, out_dir, num_workers):
    train_rows, val_rows, test_rows = load_split_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    counts = {}
    for split_name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        samples, failed = tensorize_rows(rows, num_workers)
        torch.save(samples, os.path.join(out_dir, f"{split_name}.pt"), pickle_protocol=4)
        counts[split_name] = (len(samples), failed)

    cliff_meta = {
        "test_cliff_mols": [int(r["cliff_mol"]) for r in test_rows],
        "test_smiles": [r["smiles"] for r in test_rows],
        "test_y": [float(r["y"]) for r in test_rows],
    }
    with open(os.path.join(out_dir, "cliff_metadata.json"), "w") as f:
        json.dump(cliff_meta, f)

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", required=True,
                        help="Dir of canonical pre-split CSVs (MoleculeACE 'old' folder)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Subset of CHEMBL tasks (default: all CSVs in csv_dir)")
    parser.add_argument("--workers", type=int, default=30)
    args = parser.parse_args()

    if not os.path.isdir(args.csv_dir):
        raise FileNotFoundError(args.csv_dir)

    load_vocab(args.vocab)
    mp.set_start_method("fork", force=True)

    tasks = args.tasks or discover_tasks(args.csv_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print(f"  MoleculeACE Preprocess — canonical split (van Tilborg 2022)")
    print(f"  Tasks: {len(tasks)} | val carved as first {VAL_FRAC*100:.0f}% of train")
    print("=" * 60)

    summary = []
    for task in tasks:
        csv_path = os.path.join(args.csv_dir, f"{task}.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {task}: csv not found")
            continue
        out_dir = os.path.join(args.out_dir, task)
        counts = process_task(task, csv_path, out_dir, args.workers)
        line = (
            f"  {task:<22} train={counts['train'][0]:4d} "
            f"val={counts['val'][0]:3d} test={counts['test'][0]:4d}"
        )
        if any(f for _, f in counts.values()):
            line += f"  [failed: {sum(f for _, f in counts.values())}]"
        print(line)
        summary.append((task, counts))

    # Top-level metadata (single-task regression for every CHEMBL CSV)
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump({"task_type": "regression", "num_tasks": 1, "label": "y", "tasks": tasks}, f, indent=2)

    print(f"\nDone. {len(summary)} tasks → {args.out_dir}")


if __name__ == "__main__":
    main()
