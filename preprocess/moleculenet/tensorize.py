"""MoleculeNet tensoriser: per-split CSVs -> .pt files.

Usage:
    python -m preprocess.moleculenet.tensorize \\
        --data_dir data/processed/moleculenet/random_split \\
        --vocab data/vocab/vocab_zinc.json \\
        --seeds 1 2 3 42 43 44 123 456 789 1024
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import torch

from preprocess.core.configs import DATASET_CONFIGS
from preprocess.core.utils import load_vocab, process_smiles, to_sample

DEFAULT_SEEDS = [1, 2, 3, 42, 43, 44, 123, 456, 789, 1024]
FEATURE_CACHE = {}


def load_split_csv(csv_path, dataset_name):
    cfg = DATASET_CONFIGS[dataset_name]
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], []

    if cfg["label_cols"] == "auto":
        label_cols = [c for c in rows[0].keys() if c != cfg["smiles_col"]]
    else:
        label_cols = cfg["label_cols"]

    smiles_list, labels_list = [], []
    for row in rows:
        smi = row[cfg["smiles_col"]].strip()
        if not smi:
            continue
        lab = []
        for col in label_cols:
            val = row.get(col, "").strip()
            lab.append(float("nan") if (val == "" or val.lower() == "nan") else float(val))
        smiles_list.append(smi)
        labels_list.append(lab)
    return smiles_list, labels_list


def _worker_feature(smi):
    cpu_data = process_smiles(smi)
    if cpu_data is None:
        return smi, None
    return smi, to_sample(cpu_data)


def process_split(csv_path, pt_path, dataset_name, feature_cache):
    smiles_list, labels_list = load_split_csv(csv_path, dataset_name)
    if not smiles_list:
        return 0, 0

    samples = []
    failed = 0

    for smi, labels in zip(smiles_list, labels_list):
        core = feature_cache.get(smi)
        if core is None:
            failed += 1
            continue
        sample = dict(core)
        sample["labels"] = torch.tensor(labels, dtype=torch.float32)
        samples.append(sample)

    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    torch.save(samples, pt_path, pickle_protocol=4)
    return len(samples), failed


def discover_seeds(ds_dir):
    """Fallback: enumerate seed* subdirs actually present on disk."""
    pat = re.compile(r"^seed(\d+)$")
    seeds = []
    for name in sorted(os.listdir(ds_dir)):
        m = pat.match(name)
        if m and os.path.isdir(os.path.join(ds_dir, name)):
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def collect_unique_smiles(ds_dir, dataset_name, seeds):
    """Collect unique SMILES appearing in any seed/split for a dataset directory."""
    unique = []
    seen = set()
    for seed in seeds:
        for split in ["train", "val", "test"]:
            csv_path = os.path.join(ds_dir, f"seed{seed}", f"{split}.csv")
            if not os.path.exists(csv_path):
                continue
            smiles_list, _ = load_split_csv(csv_path, dataset_name)
            for smi in smiles_list:
                if smi in seen:
                    continue
                seen.add(smi)
                unique.append(smi)
    return unique


def ensure_feature_cache(ds_name, ds_dir, seeds, pool, chunksize=64):
    """Build or extend per-dataset feature cache once, then reuse across all splits."""
    cache = FEATURE_CACHE.setdefault(ds_name, {})
    unique_smiles = collect_unique_smiles(ds_dir, ds_name, seeds)
    pending = [smi for smi in unique_smiles if smi not in cache]
    if not pending:
        print(f"  [CACHE] {ds_name}: hit {len(cache):,}")
        return cache

    print(f"  [CACHE] {ds_name}: build {len(pending):,} new / {len(unique_smiles):,} total")
    for smi, core in pool.imap(_worker_feature, pending, chunksize=chunksize):
        cache[smi] = core
    return cache


def tensorize_dataset(ds_name, ds_dir, seeds, pool, auto_discover=False, skip_existing=False):
    cfg = DATASET_CONFIGS[ds_name]
    if not os.path.isdir(ds_dir):
        print(f"  [SKIP] {ds_name}: dir not found ({ds_dir})")
        return

    if auto_discover:
        found = discover_seeds(ds_dir)
        if not found:
            print(f"  [SKIP] {ds_name}: no seed* dirs in {ds_dir}")
            return
        seeds = found

    # Pick any existing seed dir to read header for "auto" label cols
    label_cols = None
    if cfg["label_cols"] == "auto":
        for seed in seeds:
            csv_path = os.path.join(ds_dir, f"seed{seed}", "train.csv")
            if os.path.exists(csv_path):
                with open(csv_path) as f:
                    header = next(csv.reader(f))
                label_cols = [c for c in header if c != cfg["smiles_col"]]
                break
        if label_cols is None:
            print(f"  [SKIP] {ds_name}: no seed*/train.csv to infer columns")
            return
    else:
        label_cols = cfg["label_cols"]

    meta = {
        "dataset_name": ds_name,
        "task_type": cfg["task_type"],
        "label_cols": label_cols,
        "num_tasks": len(label_cols),
    }
    with open(os.path.join(ds_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"=== {ds_name} ({cfg['task_type']}, {len(label_cols)} tasks) ===")
    feature_cache = ensure_feature_cache(ds_name, ds_dir, seeds, pool)
    for seed in seeds:
        for split in ["train", "val", "test"]:
            csv_path = os.path.join(ds_dir, f"seed{seed}", f"{split}.csv")
            pt_path = os.path.join(ds_dir, f"seed{seed}", f"{split}.pt")
            if not os.path.exists(csv_path):
                continue
            if skip_existing and os.path.exists(pt_path):
                print(f"  seed{seed}/{split}: [skip — exists]")
                continue
            n_ok, n_fail = process_split(csv_path, pt_path, ds_name, feature_cache)
            print(f"  seed{seed}/{split}: {n_ok} ({n_fail} failed)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Split CSV root dir. If --mode both was used in split, pass parent dir or use --auto.")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help=f"List of random seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-discover seed* subdirs (overrides --seeds)")
    parser.add_argument("--modes", nargs="+", default=None,
                        help="If set, treat data_dir as parent containing these subdirs (e.g. random scaffold)")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip seeds whose .pt is already on disk (resume mode)")
    args = parser.parse_args()

    load_vocab(args.vocab)
    mp.set_start_method("fork", force=True)

    print(f"Vocab  : {args.vocab}")
    print(f"Data   : {args.data_dir}")
    print(f"Workers: {args.workers}\n")

    roots = [(m, os.path.join(args.data_dir, m)) for m in args.modes] if args.modes \
            else [(None, args.data_dir)]

    # Single persistent pool for the whole run — avoids 600× Pool spin-up.
    with mp.Pool(processes=args.workers) as pool:
        for tag, root in roots:
            if tag:
                print(f"\n##### Mode: {tag} (root={root}) #####\n")
            for ds_name in args.datasets:
                ds_dir = os.path.join(root, ds_name)
                tensorize_dataset(ds_name, ds_dir, args.seeds, pool,
                                  auto_discover=args.auto, skip_existing=args.skip_existing)

    print("\nDone.")


if __name__ == "__main__":
    main()
