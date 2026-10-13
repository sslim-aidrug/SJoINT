"""Split MoleculeNet CSVs into train / val / test (random or scaffold, 80/10/10).

Usage:
    python -m preprocess.moleculenet.split \\
        --mode scaffold --csv_dir data/raw/moleculenet \\
        --out_dir data/processed/moleculenet/scaffold_split \\
        --seeds 1 2 3 42 43 44 123 456 789 1024
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from preprocess.core.configs import DATASET_CONFIGS

RDLogger.DisableLog("rdApp.*")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
DEFAULT_SEEDS = [1, 2, 3, 42, 43, 44, 123, 456, 789, 1024]


def load_csv(dataset_name, csv_dir):
    cfg = DATASET_CONFIGS[dataset_name]
    with open(os.path.join(csv_dir, cfg["file"])) as f:
        rows = list(csv.DictReader(f))

    smiles_col = cfg.get("orig_smiles_col", cfg["smiles_col"])
    if cfg["label_cols"] == "auto":
        label_cols = [c for c in rows[0].keys() if c != smiles_col]
    else:
        label_cols = cfg["label_cols"]

    valid_smiles, valid_labels = [], []
    for row in rows:
        smi = row[smiles_col].strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        lab = []
        for col in label_cols:
            val = row.get(col, "").strip()
            lab.append("nan" if (val == "" or val.lower() == "nan") else val)
        valid_smiles.append(canonical)
        valid_labels.append(lab)

    return valid_smiles, valid_labels, label_cols


def get_scaffold_groups(smiles_list):
    groups = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smi
        except Exception:
            scaffold = smi
        groups[scaffold].append(idx)
    return list(groups.values())


def scaffold_split(scaffold_groups, n_total, seed):
    rng = np.random.RandomState(seed)
    groups = sorted([(len(g), rng.random(), g) for g in scaffold_groups],
                    key=lambda x: (-x[0], x[1]))
    sorted_groups = [g for _, _, g in groups]

    train_cut = int(n_total * TRAIN_RATIO)
    val_cut = int(n_total * (TRAIN_RATIO + VAL_RATIO))

    train, val, test = [], [], []
    for group in sorted_groups:
        if len(train) + len(group) <= train_cut:
            train.extend(group)
        elif len(train) + len(val) + len(group) <= val_cut:
            val.extend(group)
        else:
            test.extend(group)
    return train, val, test


def random_split(n_total, seed):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)
    train_end = int(n_total * TRAIN_RATIO)
    val_end = int(n_total * (TRAIN_RATIO + VAL_RATIO))
    return (indices[:train_end].tolist(),
            indices[train_end:val_end].tolist(),
            indices[val_end:].tolist())


def save_split_csv(filepath, smiles, labels, label_names, indices):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"] + list(label_names))
        for idx in indices:
            writer.writerow([smiles[idx]] + labels[idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["random", "scaffold", "both"],
                        help="random / scaffold / both (writes to <out_dir>/random and <out_dir>/scaffold)")
    parser.add_argument("--csv_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help=f"List of random seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()))
    args = parser.parse_args()

    modes = ["random", "scaffold"] if args.mode == "both" else [args.mode]

    print("=" * 60)
    print(f"  MoleculeNet Split — {args.mode}")
    print(f"  Datasets : {args.datasets}")
    print(f"  Seeds    : {args.seeds}")
    print("=" * 60)

    for ds in args.datasets:
        cfg = DATASET_CONFIGS[ds]
        csv_path = os.path.join(args.csv_dir, cfg["file"])
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {ds}: not found")
            continue

        smiles, labels, label_names = load_csv(ds, args.csv_dir)
        n = len(smiles)
        print(f"\n  {ds}: {n:,} valid molecules, {len(label_names)} tasks")

        scaffold_groups = get_scaffold_groups(smiles) if "scaffold" in modes else None

        for mode in modes:
            mode_out = os.path.join(args.out_dir, mode) if args.mode == "both" else args.out_dir
            for seed in args.seeds:
                if mode == "scaffold":
                    tr, va, te = scaffold_split(scaffold_groups, n, seed)
                else:
                    tr, va, te = random_split(n, seed)

                seed_dir = os.path.join(mode_out, ds, f"seed{seed}")
                save_split_csv(os.path.join(seed_dir, "train.csv"), smiles, labels, label_names, tr)
                save_split_csv(os.path.join(seed_dir, "val.csv"), smiles, labels, label_names, va)
                save_split_csv(os.path.join(seed_dir, "test.csv"), smiles, labels, label_names, te)
                tag = f"[{mode}] " if args.mode == "both" else ""
                print(f"    {tag}seed{seed}: train={len(tr)}, val={len(va)}, test={len(te)}")

    print(f"\nDone. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
