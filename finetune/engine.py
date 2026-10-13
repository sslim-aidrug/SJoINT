"""Shared fine-tuning engine: training primitives used by both the MoleculeNet
and MoleculeACE downstream tasks.

The public entry points live in `moleculenet/train.py` and `moleculeace/train.py`;
this module holds the task-agnostic pieces (seeding, metrics, the two-stage
training loop, optimiser/scheduler construction, and the model factory) so the
two tasks share one implementation.
"""
import os
import sys
import random
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

from dataset import compute_label_stats, normalize_labels
from model import (
    DownstreamModel, freeze_backbone, unfreeze_backbone,
    load_pretrained_backbone, create_random_backbone,
    is_backbone_param,
)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def _seed_worker(worker_id):
    """Seed each DataLoader worker deterministically (best-effort reproducibility)."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed, deterministic=True):
    """Seed RNGs. With `deterministic=True` (MoleculeNet) also request
    deterministic GPU kernels for run-to-run reproducibility — full determinism
    is not guaranteed (GATv2/scatter atomic ops), so warn_only falls back rather
    than erroring. With `deterministic=False` (MoleculeACE replicates / CV, where
    only the seed needs fixing, not bit-identical kernels) keep the faster
    non-deterministic cuDNN path."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
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


def _metrics_original_scale(preds, labels, label_stats, task_type):
    """For regression, undo z-score normalization before computing RMSE."""
    if task_type == "regression" and label_stats is not None:
        mean, std = label_stats
        preds = preds * std.numpy() + mean.numpy()
        labels = labels * std.numpy() + mean.numpy()
    return compute_metrics(preds, labels, task_type)


def rmse_cliff(preds, labels, cliff_mask, label_stats=None):
    """RMSE on the activity-cliff subset (original scale). preds/labels are the
    normalized (or raw) 1-D arrays; label_stats un-normalizes regression."""
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if label_stats is not None:
        m, s = label_stats
        preds = preds * s.numpy()[0] + m.numpy()[0]
        labels = labels * s.numpy()[0] + m.numpy()[0]
    mask = np.asarray(cliff_mask).astype(bool).reshape(-1)
    if mask.shape[0] != labels.shape[0] or not mask.any():
        return None, 0
    d = preds[mask] - labels[mask]
    return float(np.sqrt(np.mean(d ** 2))), int(mask.sum())


# --------------------------------------------------------------------------- #
# Batch / device
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Optimiser / scheduler
# --------------------------------------------------------------------------- #
def build_optimizer(model, args, with_backbone):
    """Stage 1: head only. Stage 2: backbone (low LR) + head (full LR).

    The prediction head trains at the full head LR in both stages; the backbone
    is off in Stage 1 and joins at `learning_rate * backbone_lr_ratio` in Stage 2.
    """
    backbone = [p for n, p in model.named_parameters() if is_backbone_param(n)]
    head = [p for n, p in model.named_parameters() if not is_backbone_param(n)]
    backbone_lr = args.learning_rate * args.backbone_lr_ratio if with_backbone else 0.0
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": backbone_lr},
            {"params": head, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )


def activate_backbone_lr(optimizer, args):
    optimizer.param_groups[0]["lr"] = args.learning_rate * args.backbone_lr_ratio


def _make_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
        cooldown=5, min_lr=1e-6)


# --------------------------------------------------------------------------- #
# Two-stage training loop
# --------------------------------------------------------------------------- #
def run_two_stage(model, train_loader, val_loader, test_loader,
                  args, device, task_type, logger, label_stats=None):
    """Two-stage training: backbone frozen -> unfreeze at stage1_epochs+1.

    `val_loader is None` selects the full-train / no-early-stop mode (train to
    `max_epochs`, keep the final model). Otherwise the best epoch is picked by
    validation loss, with optional early stopping.
    """
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
            if args.preserve_optimizer_state:
                activate_backbone_lr(optimizer, args)
            else:
                optimizer = build_optimizer(model, args, with_backbone=True)
            scheduler = _make_scheduler(optimizer)

        t_loss, t_met = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, task_type, args.grad_clip, label_stats)
        if val_loader is not None:
            v_loss, v_met = evaluate(
                model, val_loader, device, task_type, criterion, label_stats)
        else:                                    # full_train: no held-out val
            v_loss, v_met = t_loss, t_met
        if scheduler is not None:
            scheduler.step(v_loss)

        # full_train (val_loader None): keep the FINAL model (train to max_epochs)
        improved = True if val_loader is None else (v_loss < best_val_loss)
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

        if val_loader is not None and es_patience > 0 and epoch > args.stage1_epochs:
            # Measure patience from the later of (best epoch, Stage-1 end) so that
            # a good frozen-backbone (Stage-1) checkpoint cannot instantly trip
            # early-stop at Stage-2 start and defeat the fine-tuning stage.
            ref_epoch = max(best_epoch, args.stage1_epochs)
            if (epoch - ref_epoch) >= es_patience:
                logger.info(
                    f"  [Early Stop] No improvement for {es_patience} epochs "
                    f"(best epoch={best_epoch}, ref={ref_epoch})")
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


# --------------------------------------------------------------------------- #
# Model factory / misc
# --------------------------------------------------------------------------- #
def build_model(args, task_type, num_tasks, jt_feature_dim, device, logger):
    """Load (or randomly init) the backbone and attach the prediction head."""
    if args.pretrain_ckpt and os.path.exists(args.pretrain_ckpt):
        pretrained = load_pretrained_backbone(
            args.pretrain_ckpt, device,
            jt_vocab_size=args.jt_vocab_size,
            num_ca_blocks=args.num_ca_blocks)
    else:
        pretrained = create_random_backbone(
            jt_vocab_size=args.jt_vocab_size,
            num_ca_blocks=args.num_ca_blocks, device=device,
            jt_feature_dim=jt_feature_dim)
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
    return model


def normalize_regression(train_ds, *other_ds):
    """z-score regression labels using train stats; apply to every dataset given.
    Returns (mean, std). Caller must ensure datasets do not share sample objects."""
    label_mean, label_std = compute_label_stats(train_ds)
    for ds in (train_ds, *other_ds):
        normalize_labels(ds, label_mean, label_std)
    return label_mean, label_std


def infer_jt_feature_dim(dataset):
    if not dataset.data:
        raise RuntimeError("Empty dataset")
    return int(dataset.data[0]["jt_graph"].features.shape[-1])


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
