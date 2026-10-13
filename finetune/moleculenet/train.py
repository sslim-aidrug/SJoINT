"""MoleculeNet fine-tuning.
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
from utils import (                                                  # noqa: E402
    seed_everything, _seed_worker, run_two_stage, build_model,
    normalize_regression, infer_jt_feature_dim, setup_logger, to_original_scale,
)
from configs import DEFAULT_HP                                        # noqa: E402

_PROJECT_ROOT = os.path.dirname(_FINETUNE_DIR)
DEFAULT_DATA_BASE = os.path.join(_PROJECT_ROOT, "data", "processed", "moleculenet", "random_split")
DEFAULT_RESULTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_PRETRAIN_CKPT = os.path.join(_PROJECT_ROOT, "checkpoints", "SJoINT_zinc250k_pretrained.pt")


@dataclass
class Config:
    dataset: str = "BBBP"
    seed: int = 42

    num_ca_blocks: int = 3
    jt_vocab_size: int = 1750
    pretrain_ckpt: str = ""

    head_hidden: int = DEFAULT_HP["head_hidden"]
    head_dropout: float = DEFAULT_HP["head_dropout"]
    num_head_layers: int = DEFAULT_HP["num_head_layers"]
    head_activation: str = "relu"
    head_norm: str = "layer"

    learning_rate: float = DEFAULT_HP["learning_rate"]
    backbone_lr_ratio: float = DEFAULT_HP["backbone_lr_ratio"]
    weight_decay: float = DEFAULT_HP["weight_decay"]
    batch_size: int = DEFAULT_HP["batch_size"]
    max_epochs: int = 100
    stage1_epochs: int = DEFAULT_HP["stage1_epochs"]
    early_stop_patience: int = 20   # 0 disables
    grad_clip: float = 1.0
    use_amp: bool = True
    pool_mode: str = "both"
    drop_last_train: int = 1
    preserve_optimizer_state: bool = False
    backbone_dropout: float = -1.0  # -1 keeps the pre-training value
    freeze_bn: bool = False

    gpu_id: int = 0
    num_workers: int = 2
    data_base: str = ""
    results_base: str = ""
    save_best_ckpt: bool = False

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
            f"_blr{self.backbone_lr_ratio}_{self.head_norm}_{self.head_activation}_bd{self.backbone_dropout}"
            f"_bn{int(self.freeze_bn)}_po{int(self.preserve_optimizer_state)}"
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
    g.manual_seed(args.seed)
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
             preds=to_original_scale(results["test_preds"], label_stats),
             labels=to_original_scale(results["test_labels"], label_stats))
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
