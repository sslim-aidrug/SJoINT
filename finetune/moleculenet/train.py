"""MoleculeNet fine-tuning entry point.

Two-stage downstream training on the MoleculeNet ADMET benchmarks. Each dataset
ships several random/scaffold splits (one per seed); the best epoch is selected
by validation loss with early stopping.

    python finetune/moleculenet/train.py --dataset BBBP --seed 42
"""
import os
import sys
import json
import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

_FINETUNE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _FINETUNE_DIR)

from dataset import FinetuneDataset, collate_fn                       # noqa: E402
from engine import (                                                  # noqa: E402
    seed_everything, _seed_worker, run_two_stage, build_model,
    normalize_regression, infer_jt_feature_dim, setup_logger,
)

_PROJECT_ROOT = os.path.dirname(_FINETUNE_DIR)
DEFAULT_DATA_BASE = os.path.join(_PROJECT_ROOT, "data", "processed", "moleculenet", "random_split")
DEFAULT_RESULTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_PRETRAIN_CKPT = os.path.join(_PROJECT_ROOT, "checkpoints", "SJoINT_zinc250k_pretrained.pt")


@dataclass
class Config:
    # Dataset selection (one random/scaffold split per seed)
    dataset: str = "BBBP"
    seed: int = 1

    # Backbone (architecture auto-detected from the checkpoint)
    num_ca_blocks: int = 3
    jt_vocab_size: int = 1753
    pretrain_ckpt: str = ""

    # Prediction head
    head_hidden: int = 32
    head_dropout: float = 0.1
    num_head_layers: int = 2
    head_activation: str = "relu"
    head_norm: str = "layer"

    # Two-stage schedule (max_epochs / patience are fixed protocol)
    learning_rate: float = 1e-3
    backbone_lr_ratio: float = 0.1
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    stage1_epochs: int = 30
    early_stop_patience: int = 20   # 0 = disabled
    grad_clip: float = 1.0
    use_amp: bool = True
    pool_mode: str = "both"         # "both", "jt_only", "mol_only"
    drop_last_train: int = 1
    preserve_optimizer_state: bool = False

    # System / paths
    gpu_id: int = 0
    num_workers: int = 2
    data_base: str = ""
    results_base: str = ""
    save_best_ckpt: bool = False

    # Derived
    data_dir: str = ""
    output_dir: str = ""

    def __post_init__(self):
        self.pretrain_ckpt = self.pretrain_ckpt or DEFAULT_PRETRAIN_CKPT
        self.data_base = self.data_base or DEFAULT_DATA_BASE
        self.results_base = self.results_base or DEFAULT_RESULTS_BASE
        self.data_dir = os.path.join(self.data_base, self.dataset, f"seed{self.seed}")
        hp_tag = (
            f"s1e{self.stage1_epochs}_lr{self.learning_rate}_wd{self.weight_decay}_bs{self.batch_size}"
            f"_L{self.num_head_layers}_h{self.head_hidden}_d{self.head_dropout}"
        )
        self.output_dir = os.path.join(self.results_base, self.dataset, f"seed{self.seed}", hp_tag)

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="SJoINT MoleculeNet fine-tuning")
        for name, fld in cls.__dataclass_fields__.items():
            if name in {"data_dir", "output_dir"}:
                continue
            if fld.type == bool:
                parser.add_argument(f"--{name}", action="store_true", default=fld.default)
            else:
                parser.add_argument(f"--{name}", type=fld.type, default=fld.default)
        args = parser.parse_args()
        return cls(**{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__})


def build_loaders(args):
    train_ds = FinetuneDataset(os.path.join(args.data_dir, "train.pt"))
    val_ds = FinetuneDataset(os.path.join(args.data_dir, "val.pt"))
    test_ds = FinetuneDataset(os.path.join(args.data_dir, "test.pt"))
    kw = {
        "batch_size": args.batch_size, "collate_fn": collate_fn,
        "num_workers": args.num_workers, "pin_memory": True,
        "persistent_workers": args.num_workers > 0, "worker_init_fn": _seed_worker,
    }
    g = torch.Generator()
    g.manual_seed(args.seed)                       # fix the training shuffle order
    train_loader = DataLoader(train_ds, shuffle=True,
                              drop_last=bool(args.drop_last_train), generator=g, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def main():
    args = Config.from_args()
    logger = setup_logger(args.output_dir)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    with open(os.path.join(os.path.dirname(args.data_dir), "metadata.json")) as f:
        meta = json.load(f)
    task_type, num_tasks = meta["task_type"], meta["num_tasks"]

    logger.info("=" * 60)
    logger.info(f"  MoleculeNet fine-tuning: {args.dataset} ({task_type}, {num_tasks} tasks)")
    logger.info(f"  Seed {args.seed} | Device {device} | Pretrain {args.pretrain_ckpt or '(random)'}")
    logger.info(f"  LR/WD/BS {args.learning_rate}/{args.weight_decay}/{args.batch_size} | "
                f"head {args.num_head_layers}L h{args.head_hidden} d{args.head_dropout} | "
                f"max_epochs {args.max_epochs} stage1 {args.stage1_epochs} es {args.early_stop_patience}")
    logger.info("=" * 60)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(args)
    jt_feature_dim = infer_jt_feature_dim(train_ds)

    label_stats = None
    if task_type == "regression":
        mean, std = normalize_regression(train_ds, val_ds, test_ds)
        label_stats = (mean, std)
        logger.info(f"  LabelNorm mean={mean.tolist()} std={std.tolist()}")

    model = build_model(args, task_type, num_tasks, jt_feature_dim, device, logger)
    results = run_two_stage(model, train_loader, val_loader, test_loader,
                            args, device, task_type, logger, label_stats)

    out = {
        "dataset": args.dataset, "seed": args.seed,
        "task_type": task_type, "num_tasks": num_tasks,
        "best_epoch": results["best_epoch"], "best_val_loss": results["best_val_loss"],
        "test_loss": results["test_loss"], "test": results["test"],
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez(os.path.join(args.output_dir, "test_preds.npz"),
             preds=results["test_preds"], labels=results["test_labels"])
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
