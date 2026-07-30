"""MoleculeACE fine-tuning entry point (original protocol).

Trains the final model on the **full fixed training set** with the CV-selected
hyper-parameters and evaluates once on the **fixed test set**, reporting RMSE and
RMSE_cliff (RMSE on the activity-cliff subset). There is a single fixed split per
task; `--replicate` only changes the initialisation (a training replicate on the
identical data), never the split — so results are averaged over replicates, not
over data resamples.

    # hyper-parameters + epoch budget come from the CV search (moleculeace_hp.json)
    python finetune/moleculeace/train.py --task CHEMBL204_Ki --replicate 0
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from core import load_task, BackboneTemplate, make_ctx, train_eval    # noqa: E402
from engine import seed_everything, infer_jt_feature_dim, setup_logger  # noqa: E402

_FINETUNE_DIR = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_FINETUNE_DIR)
DEFAULT_DATA_BASE = os.path.join(_PROJECT_ROOT, "data", "processed", "moleculeace")
DEFAULT_PRETRAIN_CKPT = os.path.join(_PROJECT_ROOT, "checkpoints", "SJoINT_zinc250k_pretrained.pt")
DEFAULT_HP_JSON = os.path.join(_HERE, "moleculeace_hp.json")
DEFAULT_RESULTS_BASE = os.path.join(_HERE, "results")


def main():
    ap = argparse.ArgumentParser(description="MoleculeACE final training (full train, fixed test)")
    ap.add_argument("--task", required=True)
    ap.add_argument("--replicate", type=int, default=0,
                    help="training replicate index (initialisation only; data is fixed)")
    ap.add_argument("--data_base", default=DEFAULT_DATA_BASE)
    ap.add_argument("--pretrain_ckpt", default=DEFAULT_PRETRAIN_CKPT)
    ap.add_argument("--hp_json", default=DEFAULT_HP_JSON,
                    help="consolidated per-task HP + epoch budget (from CV search)")
    ap.add_argument("--max_epochs", type=int, default=0,
                    help="override the epoch budget; 0 = use the value in hp_json")
    ap.add_argument("--results_base", default=DEFAULT_RESULTS_BASE)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_best_ckpt", action="store_true")
    args = ap.parse_args()

    with open(args.hp_json) as f:
        hp_table = json.load(f)
    if args.task not in hp_table:
        raise KeyError(f"{args.task} not in {args.hp_json}")
    entry = hp_table[args.task]
    hp = entry["hp"]
    max_epochs = args.max_epochs or entry.get("epoch_budget") or entry.get("max_epochs")
    if not max_epochs:
        raise ValueError(f"no epoch budget for {args.task}; pass --max_epochs")

    output_dir = os.path.join(args.results_base, args.task, f"rep{args.replicate}")
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(os.path.join(output_dir, "results.json")):
        print(f"[skip] {args.task}/rep{args.replicate} already done", flush=True)
        return

    logger = setup_logger(output_dir)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    seed_everything(args.replicate, deterministic=False)                 # replicate = initialisation seed

    data_dir = os.path.join(args.data_base, args.task)
    train, test, cliff, meta = load_task(data_dir)
    task_type, num_tasks = meta["task_type"], meta["num_tasks"]
    jt_dim = infer_jt_feature_dim(train)

    logger.info("=" * 60)
    logger.info(f"  MoleculeACE (original protocol): {args.task} ({task_type})")
    logger.info(f"  Replicate {args.replicate} | full-train n={len(train.data)} | "
                f"test n={len(test.data)} | budget {max_epochs}ep")
    logger.info(f"  HP {entry.get('hp_tag')}")
    logger.info("=" * 60)

    template = BackboneTemplate(args.pretrain_ckpt, jt_vocab_size=1753,
                               num_ca_blocks=3, jt_feature_dim=jt_dim)
    ctx = make_ctx(task_type, num_tasks, template, device, logger)

    results = train_eval(train, test, hp, ctx, max_epochs=max_epochs,
                         early_stop_patience=0, full_train=True, cliff_mask=cliff,
                         num_workers=args.num_workers, run_seed=args.replicate,
                         output_dir=output_dir, save_best_ckpt=args.save_best_ckpt)

    out = {"task": args.task, "replicate": args.replicate,
           "task_type": task_type, "num_tasks": num_tasks,
           "epoch_budget": int(max_epochs), "hp_tag": entry.get("hp_tag"),
           "test": results["test"]}
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez(os.path.join(output_dir, "test_preds.npz"),
             preds=results["test_preds"], labels=results["test_labels"])
    logger.info(f"[done] RMSE={results['test'].get('rmse'):.4f} "
                f"RMSE_cliff={results['test'].get('rmse_cliff')} -> {output_dir}")


if __name__ == "__main__":
    main()
