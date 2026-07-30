"""MoleculeACE hyper-parameter search by k-fold cross-validation.

Following the original protocol, hyper-parameters are selected with **k-fold
cross-validation on the fixed training set** (no test-set peeking). For each
sampled HP combination we run `n_folds` CV folds — train on the other folds,
early-stop on the held-out fold — and score it by the mean held-out RMSE. The
best combination (and the median best-epoch, used later as the full-train epoch
budget) is written to `<out_dir>/<task>_hp.json`.

    python finetune/moleculeace/hp_search.py --task CHEMBL204_Ki --gpu_id 0
"""
import os
import sys
import json
import random
import logging
import argparse
import statistics

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from core import (                                                    # noqa: E402
    load_task, kfold_indices, _clone_subset, BackboneTemplate,
    make_ctx, train_eval,
)
from engine import seed_everything, infer_jt_feature_dim              # noqa: E402

_FINETUNE_DIR = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_FINETUNE_DIR)
DEFAULT_DATA_BASE = os.path.join(_PROJECT_ROOT, "data", "processed", "moleculeace")
DEFAULT_PRETRAIN_CKPT = os.path.join(_PROJECT_ROOT, "checkpoints", "SJoINT_zinc250k_pretrained.pt")

# HP search space (same grid used across the benchmark).
_SPACE = dict(
    learning_rate=[5e-4, 1e-3, 5e-3, 1e-2],
    weight_decay=[1e-5, 1e-4],
    batch_size=[32, 64, 128],
    num_head_layers=[1, 2, 3, 4],
    head_hidden=[32, 64, 128],
    head_dropout=[0.0, 0.1, 0.2, 0.3],
    stage1_epochs=[10, 20, 30, 40, 50],
)


def hp_tag(hp):
    return (f"s1e{hp['stage1_epochs']}_lr{hp['learning_rate']}_wd{hp['weight_decay']}"
            f"_bs{hp['batch_size']}_L{hp['num_head_layers']}_h{hp['head_hidden']}"
            f"_d{hp['head_dropout']}")


def sample_combos(n, rng_seed=0):
    r = random.Random(rng_seed)
    combos, seen = [], set()
    while len(combos) < n:
        hp = {k: r.choice(v) for k, v in _SPACE.items()}
        key = hp_tag(hp)
        if key in seen:
            continue
        seen.add(key)
        combos.append(hp)
    return combos


def quiet_logger():
    lg = logging.getLogger("mace_cv")
    lg.setLevel(logging.ERROR)
    if not lg.hasHandlers():
        lg.addHandler(logging.NullHandler())
    return lg


def main():
    ap = argparse.ArgumentParser(description="MoleculeACE k-fold CV HP search")
    ap.add_argument("--task", required=True)
    ap.add_argument("--data_base", default=DEFAULT_DATA_BASE)
    ap.add_argument("--pretrain_ckpt", default=DEFAULT_PRETRAIN_CKPT)
    ap.add_argument("--out_dir", default=os.path.join(_HERE, "hp_cv"))
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--cv_seed", type=int, default=42)
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--es_patience", type=int, default=12)
    ap.add_argument("--n_combos", type=int, default=24)
    ap.add_argument("--search_seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.task}_hp.json")
    if os.path.exists(out_path):
        print(f"[skip] {args.task} already done -> {out_path}", flush=True)
        return

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    seed_everything(args.cv_seed, deterministic=False)

    data_dir = os.path.join(args.data_base, args.task)
    train, _test, _cliff, meta = load_task(data_dir)
    task_type, num_tasks = meta["task_type"], meta["num_tasks"]
    jt_dim = infer_jt_feature_dim(train)
    template = BackboneTemplate(args.pretrain_ckpt, jt_vocab_size=1753,
                               num_ca_blocks=3, jt_feature_dim=jt_dim)
    logger = quiet_logger()
    ctx = make_ctx(task_type, num_tasks, template, device, logger)

    n = len(train.data)
    folds = kfold_indices(n, args.n_folds, args.cv_seed)
    combos = sample_combos(args.n_combos, args.search_seed)
    print(f"[mace-cv] {args.task}: {len(combos)} combos x {args.n_folds} folds "
          f"(n_train={n}, {task_type})", flush=True)

    results = []
    for ci, hp in enumerate(combos):
        fold_rmse, fold_ep = [], []
        for f in range(args.n_folds):
            val_idx = folds[f]
            train_idx = np.concatenate([folds[g] for g in range(args.n_folds) if g != f])
            tr_sub = _clone_subset(train, train_idx)
            va_sub = _clone_subset(train, val_idx)
            res = train_eval(tr_sub, va_sub, hp, ctx,
                             max_epochs=args.max_epochs, early_stop_patience=args.es_patience,
                             full_train=False, num_workers=args.num_workers,
                             output_dir=os.path.join(args.out_dir, "_scratch"))
            fold_rmse.append(res["test"]["rmse"])
            fold_ep.append(res["best_epoch"])
        mean_rmse = float(np.mean(fold_rmse))
        results.append({"hp": hp, "hp_tag": hp_tag(hp),
                        "cv_rmse": mean_rmse, "cv_rmse_std": float(np.std(fold_rmse)),
                        "epoch_budget": int(statistics.median(fold_ep))})
        print(f"  [{ci+1}/{len(combos)}] {hp_tag(hp)}  CV_RMSE={mean_rmse:.4f}", flush=True)

    results.sort(key=lambda r: r["cv_rmse"])
    best = results[0]
    out = {"task": args.task, "n_folds": args.n_folds, "cv_seed": args.cv_seed,
           "task_type": task_type, "selected": best, "all": results}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[mace-cv] {args.task} BEST {best['hp_tag']} "
          f"CV_RMSE={best['cv_rmse']:.4f} budget={best['epoch_budget']} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
