"""MoleculeACE hyperparameter search: GP Bayesian optimization scored by k-fold CV RMSE.
    python finetune/moleculeace/hp_search.py --task CHEMBL204_Ki --gpu_id 0
"""
import os
import sys
import json
import logging
import argparse
import statistics
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import optuna
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import (                                                    # noqa: E402
    load_task, kfold_indices, label_values, _clone_subset, BackboneTemplate,
    make_ctx, train_eval, HP_DEFAULTS,
)
from utils import seed_everything, infer_jt_feature_dim              # noqa: E402

_FINETUNE_DIR = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_FINETUNE_DIR)
DEFAULT_DATA_BASE = os.path.join(_PROJECT_ROOT, "data", "processed", "moleculeace")
DEFAULT_PRETRAIN_CKPT = os.path.join(_PROJECT_ROOT, "checkpoints", "SJoINT_zinc250k_pretrained.pt")

from moleculenet.configs import SEARCH_SPACE                          # noqa: E402


def hp_tag(hp):
    h = {**HP_DEFAULTS, **hp}
    return (f"s1e{h['stage1_epochs']}_lr{h['learning_rate']}_wd{h['weight_decay']}"
            f"_bs{h['batch_size']}_L{h['num_head_layers']}_h{h['head_hidden']}_d{h['head_dropout']}"
            f"_blr{h['backbone_lr_ratio']}_{h['head_norm']}_{h['head_activation']}_bd{h['backbone_dropout']}"
            f"_bn{int(h['freeze_bn'])}_po{int(h['preserve_optimizer_state'])}")


def suggest(trial):
    return {k: trial.suggest_categorical(k, v) for k, v in SEARCH_SPACE.items()}


def quiet_logger():
    lg = logging.getLogger("mace_cv")
    lg.setLevel(logging.ERROR)
    if not lg.hasHandlers():
        lg.addHandler(logging.NullHandler())
    return lg


_W = {}  # per-process state: data, folds, backbone template


def _init_worker(args):
    seed_everything(args.cv_seed, deterministic=False)
    train, _test, _cliff, meta = load_task(os.path.join(args.data_base, args.task))
    template = BackboneTemplate(args.pretrain_ckpt, jt_vocab_size=1750, num_ca_blocks=3,
                                jt_feature_dim=infer_jt_feature_dim(train))
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    _W.update(args=args, train=train, folds=kfold_indices(label_values(train), args.n_folds, args.cv_seed),
              ctx=make_ctx(meta["task_type"], meta["num_tasks"], template, device, quiet_logger()))


def _run_fold(job):
    """Train on all folds but f, early-stop and score on fold f; returns (val RMSE, best epoch)."""
    hp, f = job
    a, folds = _W["args"], _W["folds"]
    train_idx = np.concatenate([folds[g] for g in range(a.n_folds) if g != f])
    res = train_eval(_clone_subset(_W["train"], train_idx), _clone_subset(_W["train"], folds[f]), hp, _W["ctx"],
                     max_epochs=a.max_epochs, early_stop_patience=a.es_patience, full_train=False,
                     num_workers=a.num_workers, output_dir=os.path.join(a.out_dir, "_scratch"))
    return res["test"]["rmse"], res["best_epoch"]


def main():
    ap = argparse.ArgumentParser(description="MoleculeACE k-fold CV HP search")
    ap.add_argument("--task", required=True)
    ap.add_argument("--data_base", default=DEFAULT_DATA_BASE)
    ap.add_argument("--pretrain_ckpt", default=DEFAULT_PRETRAIN_CKPT)
    ap.add_argument("--out_dir", default=os.path.join(_HERE, "hp_cv"))
    ap.add_argument("--hp_json", default=os.path.join(_HERE, "moleculeace_hp.json"), help="hyperparameter table to update")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--cv_seed", type=int, default=42)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--es_patience", type=int, default=10)
    ap.add_argument("--n_trials", type=int, default=50, help="max combinations evaluated")
    ap.add_argument("--search_seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers per run (0 = load in the main process)")
    ap.add_argument("--fold_workers", type=int, default=1, help="processes that train the CV folds concurrently")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.task}_hp.json")
    if os.path.exists(out_path):
        print(f"[skip] {args.task} already done -> {out_path}", flush=True)
        return

    _init_worker(args)
    n, task_type = len(_W["train"].data), _W["ctx"].task_type
    print(f"[mace-cv] {args.task}: Bayesian (GP) search, <= {args.n_trials} combos x "
          f"{args.n_folds} folds (n_train={n}, {task_type})", flush=True)

    pool = None
    if args.fold_workers > 1:  # folds of one combination are independent -> train them concurrently
        pool = ProcessPoolExecutor(min(args.fold_workers, args.n_folds), mp_context=mp.get_context("spawn"),
                                   initializer=_init_worker, initargs=(args,))

    def cv_score(hp):
        jobs = [(hp, f) for f in range(args.n_folds)]
        res = list(pool.map(_run_fold, jobs)) if pool else [_run_fold(j) for j in jobs]
        fold_rmse, fold_ep = [r[0] for r in res], [r[1] for r in res]
        return {"hp": hp, "hp_tag": hp_tag(hp),
                "cv_rmse": float(np.mean(fold_rmse)), "cv_rmse_std": float(np.std(fold_rmse)),
                "median_best_epoch": int(statistics.median(fold_ep))}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    # Every finished combination is journaled, so an interrupted search resumes where it stopped.
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(
        os.path.join(args.out_dir, f"{args.task}_study.log")))
    probe = optuna.load_study(study_name=args.task, storage=storage) \
        if args.task in optuna.get_all_study_names(storage) else None
    done = [t for t in probe.trials if t.state.is_finished()] if probe else []  # NaN scores end as FAIL
    study = optuna.create_study(study_name=args.task, storage=storage, load_if_exists=True, direction="minimize",
                                sampler=optuna.samplers.GPSampler(seed=args.search_seed + len(done)))
    cache = {t.user_attrs["hp_tag"]: t.user_attrs["result"] for t in done}  # repeated proposals reuse their CV score
    if done:
        print(f"  [resume] {len(done)} combinations already evaluated", flush=True)
    for t in range(len(done), args.n_trials):
        trial = study.ask()
        hp = suggest(trial)
        tag = hp_tag(hp)
        if tag not in cache:
            cache[tag] = cv_score(hp)
            print(f"  [{t+1}/{args.n_trials}] {tag}  CV_RMSE={cache[tag]['cv_rmse']:.4f}", flush=True)
        trial.set_user_attr("hp_tag", tag)
        trial.set_user_attr("result", cache[tag])
        study.tell(trial, cache[tag]["cv_rmse"])
    if pool:
        pool.shutdown()

    results = sorted(cache.values(), key=lambda r: r["cv_rmse"])
    best = results[0]
    out = {"task": args.task, "n_folds": args.n_folds, "cv_seed": args.cv_seed,
           "search": "bayesian-gp", "n_trials": args.n_trials, "n_evaluated": len(results),
           "task_type": task_type, "selected": best, "all": results}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    table = json.load(open(args.hp_json)) if os.path.exists(args.hp_json) else {}
    table[args.task] = {"hp": best["hp"], "hp_tag": best["hp_tag"], "cv_rmse": round(best["cv_rmse"], 4)}
    with open(args.hp_json, "w") as f:
        json.dump(table, f, indent=2)
    print(f"[mace-cv] {args.task} BEST {best['hp_tag']} "
          f"CV_RMSE={best['cv_rmse']:.4f} median_best_epoch={best['median_best_epoch']} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
