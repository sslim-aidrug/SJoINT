"""MoleculeACE evaluation entry point (original protocol).

Follows the deep-learning evaluation of MoleculeACE (van Tilborg et al. 2022;
`MoleculeACE.benchmark.utils.cross_validate`): the fixed training set is split
into folds stratified by median-binarized activity; for the given `--fold`, a
model is trained on the other folds with early stopping on this fold and then
evaluated on the **fixed test set** (RMSE and RMSE_cliff). The reported value of
a task is the average over the fold models (`aggregate.py`).

    # hyper-parameters come from the CV search (moleculeace_hp.json)
    python finetune/moleculeace/train.py --task CHEMBL204_Ki --fold 0
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from core import (                                                    # noqa: E402
    load_task, kfold_indices, label_values, _clone_subset, BackboneTemplate,
    make_ctx, train_eval,
)
from engine import seed_everything, infer_jt_feature_dim, setup_logger  # noqa: E402

_FINETUNE_DIR = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_FINETUNE_DIR)
DEFAULT_DATA_BASE = os.path.join(_PROJECT_ROOT, "data", "processed", "moleculeace")
DEFAULT_PRETRAIN_CKPT = os.path.join(_PROJECT_ROOT, "checkpoints", "SJoINT_zinc250k_pretrained.pt")
DEFAULT_HP_JSON = os.path.join(_HERE, "moleculeace_hp.json")
DEFAULT_RESULTS_BASE = os.path.join(_HERE, "results")


def main():
    ap = argparse.ArgumentParser(description="MoleculeACE per-fold training + fixed-test evaluation")
    ap.add_argument("--task", required=True)
    ap.add_argument("--fold", type=int, default=0, help="validation fold index (0 .. n_folds-1)")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--cv_seed", type=int, default=42)
    ap.add_argument("--data_base", default=DEFAULT_DATA_BASE)
    ap.add_argument("--pretrain_ckpt", default=DEFAULT_PRETRAIN_CKPT)
    ap.add_argument("--hp_json", default=DEFAULT_HP_JSON, help="HP table: per-task entries (from the CV search) and a 'default' entry")
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--es_patience", type=int, default=10)
    ap.add_argument("--results_base", default=DEFAULT_RESULTS_BASE)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_best_ckpt", action="store_true")
    args = ap.parse_args()

    with open(args.hp_json) as f:
        hp_table = json.load(f)
    entry = hp_table[args.task] if args.task in hp_table else hp_table["default"]
    hp = entry["hp"]

    output_dir = os.path.join(args.results_base, args.task, f"fold{args.fold}")
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(os.path.join(output_dir, "results.json")):
        print(f"[skip] {args.task}/fold{args.fold} already done", flush=True)
        return

    logger = setup_logger(output_dir)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    seed_everything(args.cv_seed + args.fold, deterministic=False)

    data_dir = os.path.join(args.data_base, args.task)
    train, test, cliff, meta = load_task(data_dir)
    task_type, num_tasks = meta["task_type"], meta["num_tasks"]
    jt_dim = infer_jt_feature_dim(train)

    folds = kfold_indices(label_values(train), args.n_folds, args.cv_seed)
    val_idx = folds[args.fold]
    tr_idx = np.concatenate([folds[g] for g in range(args.n_folds) if g != args.fold])
    tr_sub, va_sub = _clone_subset(train, tr_idx), _clone_subset(train, val_idx)
    te_sub = _clone_subset(test, np.arange(len(test.data)))

    logger.info("=" * 60)
    logger.info(f"  MoleculeACE (original protocol): {args.task} ({task_type})")
    logger.info(f"  Fold {args.fold}/{args.n_folds} | train n={len(tr_sub.data)} | "
                f"val n={len(va_sub.data)} | test n={len(te_sub.data)}")
    logger.info(f"  HP {entry.get('hp_tag')}")
    logger.info("=" * 60)

    template = BackboneTemplate(args.pretrain_ckpt, jt_vocab_size=1750,
                               num_ca_blocks=3, jt_feature_dim=jt_dim)
    ctx = make_ctx(task_type, num_tasks, template, device, logger)

    results = train_eval(tr_sub, va_sub, hp, ctx, max_epochs=args.max_epochs,
                         early_stop_patience=args.es_patience, full_train=False,
                         cliff_mask=cliff, num_workers=args.num_workers,
                         run_seed=args.cv_seed + args.fold, output_dir=output_dir,
                         save_best_ckpt=args.save_best_ckpt, test_sub=te_sub)

    out = {"task": args.task, "fold": args.fold, "n_folds": args.n_folds,
           "cv_seed": args.cv_seed, "task_type": task_type, "num_tasks": num_tasks,
           "best_epoch": results.get("best_epoch"), "hp_tag": entry.get("hp_tag"),
           "test": results["test"]}
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez(os.path.join(output_dir, "test_preds.npz"),
             preds=results["test_preds"], labels=results["test_labels"])
    logger.info(f"[done] RMSE={results['test'].get('rmse'):.4f} "
                f"RMSE_cliff={results['test'].get('rmse_cliff')} -> {output_dir}")


if __name__ == "__main__":
    main()
