"""Shared fine-tuning utilities: seeding, metrics, two-stage training loop, model factory."""
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


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed, deterministic=True):
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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics(preds, labels, task_type):
    """ROC-AUC (classification) or RMSE (regression), averaged over tasks."""
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


def to_original_scale(arr, label_stats):
    if label_stats is None:
        return arr
    mean, std = label_stats
    return arr * std.numpy() + mean.numpy()


def _metrics_original_scale(preds, labels, label_stats, task_type):
    if task_type == "regression":
        preds = to_original_scale(preds, label_stats)
        labels = to_original_scale(labels, label_stats)
    return compute_metrics(preds, labels, task_type)


def rmse_cliff(preds, labels, cliff_mask, label_stats=None):
    """RMSE over the activity-cliff compounds, in original units."""
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


def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    device, task_type, grad_clip=1.0, label_stats=None, freeze_bn=False):
    model.train()
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
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
    total_loss, n_labels = 0.0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        if batch is None:
            continue
        jt_b, mol_b, labels, masks = _to_device(batch, device)

        with autocast(device_type="cuda", enabled=True):
            logits = model(jt_b, mol_b, *masks)
            mask = ~torch.isnan(labels)
            n = int(mask.sum())
            if n > 0:
                total_loss += criterion(logits[mask], labels[mask]).item() * n
                n_labels += n

        all_preds.append(logits.float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds_cat = np.concatenate(all_preds)
    labels_cat = np.concatenate(all_labels)
    metrics = _metrics_original_scale(preds_cat, labels_cat, label_stats, task_type)
    loss = total_loss / max(n_labels, 1)
    if return_preds:
        return loss, metrics, preds_cat, labels_cat
    return loss, metrics


def build_optimizer(model, args, with_backbone):
    """AdamW with the head at the full LR and the backbone at learning_rate * backbone_lr_ratio (or off)."""
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


def run_two_stage(model, train_loader, val_loader, test_loader,
                  args, device, task_type, logger, label_stats=None):
    """Stage 1 trains the head only; stage 2 unfreezes the backbone. Best epoch by validation loss
    (or the final epoch when val_loader is None)."""
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
            device, task_type, args.grad_clip, label_stats,
            freeze_bn=getattr(args, "freeze_bn", False))
        if val_loader is not None:
            v_loss, v_met = evaluate(
                model, val_loader, device, task_type, criterion, label_stats)
        else:
            v_loss, v_met = t_loss, t_met
        if scheduler is not None:
            scheduler.step(v_loss)

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
            ref_epoch = max(best_epoch, args.stage1_epochs)  # patience counted from the stage-2 start at the earliest
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


def build_model(args, task_type, num_tasks, jt_feature_dim, device, logger):
    """Backbone (pre-trained, or random when pretrain_ckpt == "none") + prediction head, backbone frozen."""
    if args.pretrain_ckpt != "none":
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
    set_backbone_dropout(model, getattr(args, "backbone_dropout", -1.0))
    freeze_backbone(model)
    return model


def set_backbone_dropout(model, p):
    """Override every backbone dropout with p (p < 0 keeps the pre-training values)."""
    if p < 0:
        return
    for mod in (model.mol_encoder, model.jt_encoder, *model.blocks):
        for m in mod.modules():
            if isinstance(m, nn.Dropout):
                m.p = p
        if hasattr(mod, "attn_dropout_p"):
            mod.attn_dropout_p = p


def normalize_regression(train_ds, *other_ds):
    """z-score all given datasets with the training-set label statistics; returns (mean, std)."""
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
