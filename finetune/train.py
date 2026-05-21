"""Two-stage fine-tuning: config, training loop, and entry point."""
import os
import sys
import json
import logging
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from dataset import (
    FinetuneDataset, collate_fn,
    compute_label_stats, normalize_labels,
)
from model import (
    DownstreamModel, freeze_backbone, unfreeze_backbone,
    load_pretrained_backbone, create_random_backbone,
)


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

DEFAULT_DATA_BASE = os.path.join(_PROJECT_ROOT, "data", "moleculenet", "random_split")
DEFAULT_RESULTS_BASE = os.path.join(_SCRIPT_DIR, "results")
DEFAULT_PRETRAIN_CKPT = ""


@dataclass
class ExperimentConfig:
    # Dataset selection
    dataset: str = "BBBP"
    seed: int = 1

    # Backbone architecture (must match the loaded pretrain checkpoint)
    num_ca_blocks: int = 3
    jt_vocab_size: int = 1753

    # Pretrain checkpoint
    pretrain_ckpt: str = ""

    # Variant tag for output path
    variant: str = ""

    # Prediction head
    head_hidden: int = 32
    head_dropout: float = 0.1
    num_head_layers: int = 2
    head_activation: str = "relu"
    head_norm: str = "layer"

    # Two-stage fine-tuning schedule
    learning_rate: float = 1e-3
    backbone_lr_ratio: float = 0.1
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    stage1_epochs: int = 30
    early_stop_patience: int = 40   # 0 = disabled
    grad_clip: float = 1.0
    use_amp: bool = True
    pool_mode: str = "both"  # "both", "jt_only", "mol_only"

    # System
    gpu_id: int = 0
    num_workers: int = 2

    # Paths
    data_base: str = ""
    results_base: str = ""

    save_best_ckpt: bool = False

    # Derived (filled in __post_init__)
    data_dir: str = ""
    output_dir: str = ""

    def __post_init__(self):
        if not self.pretrain_ckpt:
            self.pretrain_ckpt = DEFAULT_PRETRAIN_CKPT
        if not self.data_base:
            self.data_base = DEFAULT_DATA_BASE
        if not self.results_base:
            self.results_base = DEFAULT_RESULTS_BASE
        self.data_dir = os.path.join(self.data_base, self.dataset, f"seed{self.seed}")
        hp_tag = (
            f"s1e{self.stage1_epochs}_lr{self.learning_rate}_wd{self.weight_decay}_bs{self.batch_size}"
            f"_L{self.num_head_layers}_h{self.head_hidden}_d{self.head_dropout}"
        )
        parts = [self.results_base]
        if self.variant:
            parts.append(self.variant)
        parts += [self.dataset, f"seed{self.seed}", hp_tag]
        self.output_dir = os.path.join(*parts)

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="SJoINT two-stage fine-tuning")
        derived = {"data_dir", "output_dir"}
        for name, fld in cls.__dataclass_fields__.items():
            if name in derived:
                continue
            if fld.type == bool:
                parser.add_argument(f"--{name}", action="store_true", default=fld.default)
            else:
                parser.add_argument(f"--{name}", type=fld.type, default=fld.default)
        args = parser.parse_args()
        return cls(**{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__})


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics(preds, labels, task_type):
    """Compute task-appropriate metric, averaged over output dimensions."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    if task_type == "classification":
        scores = []
        for i in range(labels.shape[1]):
            mask = ~np.isnan(labels[:, i])
            if mask.sum() < 2:
                continue
            y_true = labels[mask, i]
            if len(np.unique(y_true)) < 2:
                continue
            y_score = _sigmoid(preds[mask, i])
            try:
                scores.append(roc_auc_score(y_true, y_score))
            except ValueError:
                continue
        return {"roc_auc": float(np.mean(scores)) if scores else 0.0}

    rmses = []
    for i in range(labels.shape[1]):
        mask = ~np.isnan(labels[:, i])
        if mask.sum() == 0:
            continue
        diff = preds[mask, i] - labels[mask, i]
        rmses.append(float(np.sqrt(np.mean(diff ** 2))))
    return {"rmse": float(np.mean(rmses)) if rmses else float("inf")}


_MASK_KEYS = (
    "jt2mol_b", "jt2mol_r", "jt2mol_c",
    "mol2jt_b", "mol2jt_r", "mol2jt_c",
)


def _to_device(batch, device):
    jt_b = batch["jt_batch"].to(device, non_blocking=True)
    mol_b = batch["mol_batch"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    masks = tuple(batch[k].to(device, non_blocking=True) for k in _MASK_KEYS)
    return jt_b, mol_b, labels, masks


def _metrics_original_scale(preds, labels, label_stats, task_type):
    """For regression, undo z-score normalization before computing RMSE."""
    if task_type == "regression" and label_stats is not None:
        mean, std = label_stats
        preds = preds * std.numpy() + mean.numpy()
        labels = labels * std.numpy() + mean.numpy()
    return compute_metrics(preds, labels, task_type)


def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    device, task_type, grad_clip=1.0, label_stats=None):
    model.train()
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        if batch is None:
            continue
        jt_b, mol_b, labels, masks = _to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=True):
            logits = model(jt_b, mol_b, *masks)
            mask = ~torch.isnan(labels)
            if mask.sum() == 0:
                continue
            loss = criterion(logits[mask], labels[mask])

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(logits.float().detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    metrics = _metrics_original_scale(
        np.concatenate(all_preds), np.concatenate(all_labels),
        label_stats, task_type)
    return total_loss / max(n_batches, 1), metrics


@torch.no_grad()
def evaluate(model, loader, device, task_type, criterion, label_stats=None,
             return_preds=False):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        if batch is None:
            continue
        jt_b, mol_b, labels, masks = _to_device(batch, device)

        with autocast(device_type="cuda", enabled=True):
            logits = model(jt_b, mol_b, *masks)
            mask = ~torch.isnan(labels)
            if mask.sum() > 0:
                total_loss += criterion(logits[mask], labels[mask]).item()
                n_batches += 1

        all_preds.append(logits.float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds_cat = np.concatenate(all_preds)
    labels_cat = np.concatenate(all_labels)
    metrics = _metrics_original_scale(preds_cat, labels_cat, label_stats, task_type)
    if return_preds:
        return total_loss / max(n_batches, 1), metrics, preds_cat, labels_cat
    return total_loss / max(n_batches, 1), metrics


def build_optimizer(model, args, with_backbone):
    """Stage 1: head only. Stage 2: backbone (low LR) + head (full LR)."""
    if with_backbone:
        backbone = [p for n, p in model.named_parameters()
                    if not n.startswith("pred_head")]
        head = [p for n, p in model.named_parameters()
                if n.startswith("pred_head")]
        return torch.optim.AdamW(
            [
                {"params": backbone, "lr": args.learning_rate * args.backbone_lr_ratio},
                {"params": head, "lr": args.learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay)


def _make_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
        cooldown=5, min_lr=1e-6)


def run_two_stage(model, train_loader, val_loader, test_loader,
                  args, device, task_type, logger, label_stats=None):
    """Two-stage training: backbone frozen -> unfreeze at stage1_epochs+1."""
    criterion = nn.BCEWithLogitsLoss() if task_type == "classification" else nn.MSELoss()
    metric = "roc_auc" if task_type == "classification" else "rmse"

    optimizer = build_optimizer(model, args, with_backbone=False)
    scheduler = None
    scaler = GradScaler(enabled=args.use_amp)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    es_patience = args.early_stop_patience

    for epoch in range(1, args.max_epochs + 1):
        if epoch == args.stage1_epochs + 1:
            logger.info(f"  [Stage 2] Unfreezing backbone at epoch {epoch}")
            unfreeze_backbone(model)
            optimizer = build_optimizer(model, args, with_backbone=True)
            scheduler = _make_scheduler(optimizer)

        t_loss, t_met = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, task_type, args.grad_clip, label_stats)
        v_loss, v_met = evaluate(
            model, val_loader, device, task_type, criterion, label_stats)
        if scheduler is not None:
            scheduler.step(v_loss)

        improved = v_loss < best_val_loss
        if improved:
            best_val_loss = v_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        lr = optimizer.param_groups[-1]["lr"]
        marker = " *" if improved else ""
        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {t_loss:.4f} | Train {metric}: {t_met[metric]:.4f} | "
            f"Val Loss: {v_loss:.4f} | Val {metric}: {v_met[metric]:.4f} | "
            f"LR: {lr:.2e}{marker}"
        )

        if es_patience > 0 and epoch > args.stage1_epochs:
            if (epoch - best_epoch) >= es_patience:
                logger.info(
                    f"  [Early Stop] No improvement for {es_patience} epochs "
                    f"(best epoch={best_epoch})")
                break

    if best_state is not None:
        if getattr(args, "save_best_ckpt", False):
            ckpt_path = os.path.join(args.output_dir, "best_model.ckpt")
            torch.save(best_state, ckpt_path)
            logger.info(f"  [Best ckpt] saved to {ckpt_path}")
        model.load_state_dict(best_state)
        model.to(device)

    test_loss, test_met, test_preds, test_labels = evaluate(
        model, test_loader, device, task_type, criterion, label_stats,
        return_preds=True)
    logger.info(
        f"[Final] Best epoch: {best_epoch} | "
        f"Best Val Loss: {best_val_loss:.4f} | "
        f"Test Loss: {test_loss:.4f} | Test {metric}: {test_met[metric]:.4f}"
    )

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test": test_met,
        "test_preds": test_preds,
        "test_labels": test_labels,
    }


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(output_dir)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(os.path.join(output_dir, "downstream.log"), mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def build_loaders(args):
    train_ds = FinetuneDataset(os.path.join(args.data_dir, "train.pt"))
    val_ds = FinetuneDataset(os.path.join(args.data_dir, "val.pt"))
    test_ds = FinetuneDataset(os.path.join(args.data_dir, "test.pt"))

    kw = {
        "batch_size": args.batch_size,
        "collate_fn": collate_fn,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def main():
    args = ExperimentConfig.from_args()
    logger = setup_logger(args.output_dir)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    meta_path = os.path.join(os.path.dirname(args.data_dir), "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    task_type = meta["task_type"]
    num_tasks = meta["num_tasks"]

    logger.info("=" * 60)
    logger.info("  Two-stage Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"  Dataset   : {args.dataset} ({task_type}, {num_tasks} tasks)")
    logger.info(f"  Seed      : {args.seed}")
    logger.info(f"  Device    : {device}")
    logger.info(f"  Pretrain  : {args.pretrain_ckpt or '(random init)'}")
    logger.info(f"  LR/WD/BS  : {args.learning_rate} / {args.weight_decay} / {args.batch_size}")
    logger.info(f"  Head      : {args.num_head_layers} layers, hidden={args.head_hidden}, "
                f"dropout={args.head_dropout}")
    logger.info(f"  Schedule  : max_epochs={args.max_epochs}, stage1={args.stage1_epochs}, "
                f"early_stop={args.early_stop_patience}")
    logger.info("=" * 60)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(args)

    label_stats = None
    if task_type == "regression":
        label_mean, label_std = compute_label_stats(train_ds)
        for ds in (train_ds, val_ds, test_ds):
            normalize_labels(ds, label_mean, label_std)
        label_stats = (label_mean, label_std)
        logger.info(
            f"  LabelNorm : mean={label_mean.tolist()}, std={label_std.tolist()}")

    if args.pretrain_ckpt and os.path.exists(args.pretrain_ckpt):
        pretrained = load_pretrained_backbone(
            args.pretrain_ckpt, device,
            jt_vocab_size=args.jt_vocab_size,
            num_ca_blocks=args.num_ca_blocks)
    else:
        pretrained = create_random_backbone(
            jt_vocab_size=args.jt_vocab_size,
            num_ca_blocks=args.num_ca_blocks, device=device)
        logger.info("  [No pretrain] Random initialization")
    model = DownstreamModel(
        pretrained, num_tasks,
        num_head_layers=args.num_head_layers,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        head_activation=args.head_activation,
        head_norm=args.head_norm,
        pool_mode=args.pool_mode,
    ).to(device)
    del pretrained

    freeze_backbone(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Params    : {n_params:,} (Stage 1 trainable: {n_trainable:,})")

    logger.info("=" * 60)

    results = run_two_stage(
        model, train_loader, val_loader, test_loader,
        args, device, task_type, logger, label_stats)

    out = {
        "dataset": args.dataset,
        "seed": args.seed,
        "task_type": task_type,
        "num_tasks": num_tasks,
        "best_epoch": results["best_epoch"],
        "best_val_loss": results["best_val_loss"],
        "test_loss": results["test_loss"],
        "test": results["test"],
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Save test predictions
    if "test_preds" in results:
        np.savez(os.path.join(args.output_dir, "test_preds.npz"),
                 preds=results["test_preds"],
                 labels=results["test_labels"])
        logger.info(f"Test predictions saved to {args.output_dir}/test_preds.npz")

    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
