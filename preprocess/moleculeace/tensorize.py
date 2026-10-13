"""MoleculeACE tensoriser: per-task activity-cliff CSVs -> .pt.

Following the original protocol (van Tilborg et al. 2022), each of the 30 CHEMBL
tasks has a single **fixed train/test split**. Per task dir: `train.csv`
(smiles,y) and `test.csv` (smiles,y,cliff_mol). This writes `train.pt` /
`test.pt` (FinetuneDataset format) plus `cliff_mask.pt` (per-test-molecule
activity-cliff 0/1 flags) and a task-level `metadata.json`.

The MoleculeACE benchmark is from van Tilborg et al. 2022
(https://github.com/molML/MoleculeACE); place the raw CSVs under
`data/raw/moleculeace/<task>/{train,test}.csv`.

Usage:
    python -m preprocess.moleculeace.tensorize \\
        --data_dir data/raw/moleculeace \\
        --out_dir data/processed/moleculeace \\
        --vocab data/vocab/vocab_zinc.json --workers 30
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import multiprocessing as mp

import torch

from preprocess.core.utils import load_vocab, process_smiles, to_sample


def _sample(args):
    smi, y = args
    cpu = process_smiles(smi)
    if cpu is None:
        return None
    return to_sample(cpu, labels=[float(y)])          # jt_features already 61-D


def _read_csv(path, with_cliff=False):
    rows = list(csv.DictReader(open(path)))
    data = [(r["smiles"].strip(), r["y"]) for r in rows]
    cliff = [int(float(r.get("cliff_mol", 0))) for r in rows] if with_cliff else None
    return data, cliff


def _tensorize(path, pool, with_cliff=False):
    data, cliff = _read_csv(path, with_cliff)
    samples = pool.map(_sample, data)
    out, mask = [], []
    for s, c in zip(samples, (cliff or [0] * len(samples))):
        if s is None:
            continue
        out.append(s); mask.append(c)
    return out, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="raw CSV root: <task>/{train,test}.csv")
    ap.add_argument("--out_dir", required=True, help="output .pt root")
    ap.add_argument("--vocab", required=True, help="JT vocab JSON")
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()

    load_vocab(args.vocab)
    tasks = sorted(d for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d)))
    print(f"[moleculeace] {len(tasks)} tasks -> {args.out_dir}", flush=True)
    pool = mp.Pool(processes=args.workers)
    for task in tasks:
        out = os.path.join(args.out_dir, task)
        os.makedirs(out, exist_ok=True)
        tr, _ = _tensorize(os.path.join(args.data_dir, task, "train.csv"), pool)
        te, mask = _tensorize(os.path.join(args.data_dir, task, "test.csv"), pool, with_cliff=True)
        torch.save(tr, os.path.join(out, "train.pt"))
        torch.save(te, os.path.join(out, "test.pt"))
        torch.save(torch.tensor(mask, dtype=torch.long), os.path.join(out, "cliff_mask.pt"))
        json.dump({"dataset_name": task, "task_type": "regression",
                   "label_cols": ["y"], "num_tasks": 1,
                   "n_train": len(tr), "n_test": len(te), "n_cliff": int(sum(mask))},
                  open(os.path.join(out, "metadata.json"), "w"), indent=2)
        print(f"  {task}: train {len(tr)}, test {len(te)} (cliff {int(sum(mask))})", flush=True)
    pool.close(); pool.join()
    print("[moleculeace] all done", flush=True)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
