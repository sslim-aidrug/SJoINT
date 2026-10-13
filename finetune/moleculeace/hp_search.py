"""MoleculeACE hyper-parameter search by k-fold cross-validation.

Following the original protocol (van Tilborg et al., 2022), hyper-parameters are
selected with **k-fold cross-validation on the fixed training set** (folds
stratified by median-binarized activity; no test-set peeking) and **Bayesian
optimization with a Gaussian process** (Optuna GPSampler),
evaluating at most `n_trials` (default 50) combinations of the search space. Each
proposed combination is scored by the mean held-out RMSE over `n_folds` CV folds
(train on the other folds, early-stop on the held-out fold); a combination the
optimizer proposes again reuses its cached score. The best combination is written
to `<out_dir>/<task>_hp.json`.

    python finetune/moleculeace/hp_search.py --task CHEMBL204_Ki --gpu_id 0
"""
import os
import sys
import json
import logging
import argparse
import statistics
import warnings

import numpy as np
import optuna
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from core import (                                                    # noqa: E402
    load_task, kfold_indices, label_values, _clone_subset, BackboneTemplate,
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


def suggest(trial):
    """One HP combination from the (categorical) search space."""
    return {k: trial.suggest_categorical(k, v) for k, v in _SPACE.items()}


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
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--es_patience", type=int, default=10)
    ap.add_argument("--n_trials", type=int, default=50,
                    help="max HP combinations evaluated by Bayesian optimization")
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
    folds = kfold_indices(label_values(train), args.n_folds, args.cv_seed)
    print(f"[mace-cv] {args.task}: Bayesian (GP) search, <= {args.n_trials} combos x "
          f"{args.n_folds} folds (n_train={n}, {task_type})", flush=True)

    def cv_score(hp):
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
        return {"hp": hp, "hp_tag": hp_tag(hp),
                "cv_rmse": float(np.mean(fold_rmse)), "cv_rmse_std": float(np.std(fold_rmse)),
                "median_best_epoch": int(statistics.median(fold_ep))}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)  # GPSampler
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.GPSampler(seed=args.search_seed))
    cache = {}                                    # hp_tag -> CV result (repeat proposals reuse it)
    for t in range(args.n_trials):
        trial = study.ask()
        hp = suggest(trial)
        tag = hp_tag(hp)
        if tag not in cache:
            cache[tag] = cv_score(hp)
            print(f"  [{t+1}/{args.n_trials}] {tag}  CV_RMSE={cache[tag]['cv_rmse']:.4f}", flush=True)
        study.tell(trial, cache[tag]["cv_rmse"])

    results = sorted(cache.values(), key=lambda r: r["cv_rmse"])
    best = results[0]
    out = {"task": args.task, "n_folds": args.n_folds, "cv_seed": args.cv_seed,
           "search": "bayesian-gp", "n_trials": args.n_trials, "n_evaluated": len(results),
           "task_type": task_type, "selected": best, "all": results}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[mace-cv] {args.task} BEST {best['hp_tag']} "
          f"CV_RMSE={best['cv_rmse']:.4f} median_best_epoch={best['median_best_epoch']} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
