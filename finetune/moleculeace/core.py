"""Shared MoleculeACE helpers: fixed-split data loading, stratified k-fold
construction, a reusable backbone template, and a single train/evaluate call.

MoleculeACE follows the original deep-learning protocol (van Tilborg et al. 2022;
`MoleculeACE.benchmark.utils.cross_validate`): one **fixed train/test split** per
task; the training set is split into **5 folds stratified by median-binarized
activity**; hyper-parameters are chosen by Bayesian optimization scored with this
cross-validation; and for evaluation one model is trained per fold (4 folds for
training, 1 for early stopping), each model predicts the fixed test set, and the
test metrics are **averaged over the 5 fold models**.
"""
import os
import sys
import copy
import json
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# Many single-task processes share the GPU; cap CPU threads per process to avoid
# oversubscription across the pool (which otherwise starves every process).
torch.set_num_threads(2)

_FINETUNE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _FINETUNE_DIR)

from dataset import (                                                 # noqa: E402
    FinetuneDataset, collate_fn, compute_label_stats, normalize_labels,
)
from model import DownstreamModel, freeze_backbone                    # noqa: E402
from engine import (                                                  # noqa: E402
    _seed_worker, run_two_stage, rmse_cliff, infer_jt_feature_dim, to_original_scale,
)
from model import load_pretrained_backbone, create_random_backbone    # noqa: E402


# --------------------------------------------------------------------------- #
# Fixed-split data
# --------------------------------------------------------------------------- #
def load_task(data_dir):
    """Load a task's fixed split. `data_dir` holds train.pt / test.pt /
    cliff_mask.pt / metadata.json (no per-seed subdirectory)."""
    train = FinetuneDataset(os.path.join(data_dir, "train.pt"))
    test = FinetuneDataset(os.path.join(data_dir, "test.pt"))
    with open(os.path.join(data_dir, "metadata.json")) as f:
        meta = json.load(f)
    cliff_path = os.path.join(data_dir, "cliff_mask.pt")
    cliff = None
    if os.path.exists(cliff_path):
        cliff = torch.load(cliff_path, map_location="cpu", weights_only=False).numpy()
    return train, test, cliff, meta


def _clone_subset(dataset, indices):
    """A shallow FinetuneDataset holding deep-copied samples at `indices`
    (deep-copied so per-fold label normalisation cannot cross-contaminate)."""
    sub = FinetuneDataset.__new__(FinetuneDataset)
    sub.data = [copy.deepcopy(dataset.data[i]) for i in indices]
    return sub


def label_values(dataset):
    """First-task label of every sample (MoleculeACE tasks are single-task)."""
    return np.array([float(s["labels"].view(-1)[0]) for s in dataset.data])


def kfold_indices(y, n_folds, cv_seed):
    """Validation-fold indices of a k-fold split stratified by median-binarized
    activity, as in MoleculeACE (`StratifiedKFold(shuffle=True)`, fixed seed)."""
    y = np.asarray(y, dtype=float)
    labels = (y >= np.median(y)).astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
    return [val for _, val in skf.split(np.zeros(len(y)), labels)]


# --------------------------------------------------------------------------- #
# Backbone template (load once, reuse across many runs)
# --------------------------------------------------------------------------- #
class BackboneTemplate:
    """Loads the pretrained backbone once; hands out fresh CPU copies so each
    run starts from the same pretrained weights without re-reading the file."""

    def __init__(self, pretrain_ckpt, jt_vocab_size=1750, num_ca_blocks=3,
                 jt_feature_dim=61):
        self.pretrain_ckpt = pretrain_ckpt
        if pretrain_ckpt != "none":
            bb = load_pretrained_backbone(pretrain_ckpt, "cpu",
                                          jt_vocab_size=jt_vocab_size,
                                          num_ca_blocks=num_ca_blocks)
            self.pretrained = True
        else:
            bb = create_random_backbone(jt_vocab_size=jt_vocab_size,
                                        num_ca_blocks=num_ca_blocks, device="cpu",
                                        jt_feature_dim=jt_feature_dim)
            self.pretrained = False
        self._template = bb.cpu()

    def fresh(self):
        return copy.deepcopy(self._template)


# --------------------------------------------------------------------------- #
# One train/evaluate call
# --------------------------------------------------------------------------- #
_ENGINE_DEFAULTS = dict(
    backbone_lr_ratio=0.1, grad_clip=1.0, use_amp=True,
    preserve_optimizer_state=False, pool_mode="both",
    head_activation="relu", head_norm="layer", save_best_ckpt=False,
)


def _engine_args(hp, max_epochs, early_stop_patience, output_dir, save_best_ckpt):
    return SimpleNamespace(
        learning_rate=hp["learning_rate"], weight_decay=hp["weight_decay"],
        batch_size=hp["batch_size"], stage1_epochs=hp["stage1_epochs"],
        max_epochs=max_epochs, early_stop_patience=early_stop_patience,
        num_head_layers=hp["num_head_layers"], head_hidden=hp["head_hidden"],
        head_dropout=hp["head_dropout"], output_dir=output_dir,
        **{k: v for k, v in _ENGINE_DEFAULTS.items() if k != "save_best_ckpt"},
        save_best_ckpt=save_best_ckpt,
    )


def _loader(data_list, batch_size, num_workers, shuffle, seed=0, drop_last=False):
    ds = FinetuneDataset.__new__(FinetuneDataset)
    ds.data = data_list
    kw = dict(batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers,
              pin_memory=True, persistent_workers=num_workers > 0,
              worker_init_fn=_seed_worker)
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(ds, shuffle=True, drop_last=drop_last, generator=g, **kw)
    return DataLoader(ds, shuffle=False, **kw)


def train_eval(train_sub, eval_sub, hp, ctx, *, max_epochs, early_stop_patience,
               full_train, cliff_mask=None, num_workers=2, run_seed=0,
               output_dir="/tmp", save_best_ckpt=False, test_sub=None):
    """Build a fresh model from the backbone template, run two-stage training on
    `train_sub`, and evaluate on `eval_sub`. `full_train=True` uses no held-out
    val (train to max_epochs, keep final model); otherwise `eval_sub` is the val
    set with early stopping. If `test_sub` is given, the model selected on
    `eval_sub` is evaluated on `test_sub` instead. Returns the results dict
    (+ rmse_cliff if a mask is given). Regression labels are z-scored using the
    training subset's stats."""
    task_type, num_tasks = ctx.task_type, ctx.num_tasks

    label_stats = None
    if task_type == "regression":
        mean, std = compute_label_stats(train_sub)
        normalize_labels(train_sub, mean, std)
        if eval_sub is not train_sub:
            normalize_labels(eval_sub, mean, std)
        if test_sub is not None:
            normalize_labels(test_sub, mean, std)
        label_stats = (mean, std)

    bb = ctx.template.fresh().to(ctx.device)
    model = DownstreamModel(
        bb, num_tasks, num_head_layers=hp["num_head_layers"],
        head_hidden=hp["head_hidden"], head_dropout=hp["head_dropout"],
        head_activation="relu", head_norm="layer", pool_mode="both").to(ctx.device)
    del bb
    freeze_backbone(model)

    args = _engine_args(hp, max_epochs, early_stop_patience, output_dir, save_best_ckpt)
    os.makedirs(output_dir, exist_ok=True)

    train_loader = _loader(train_sub.data, hp["batch_size"], num_workers,
                           shuffle=True, seed=run_seed, drop_last=True)
    eval_loader = _loader(eval_sub.data, hp["batch_size"], num_workers, shuffle=False)
    val_loader = None if full_train else eval_loader
    test_loader = eval_loader if test_sub is None else \
        _loader(test_sub.data, hp["batch_size"], num_workers, shuffle=False)

    results = run_two_stage(model, train_loader, val_loader, test_loader,
                            args, ctx.device, task_type, ctx.logger, label_stats)

    if cliff_mask is not None and task_type == "regression":
        rc, n = rmse_cliff(results["test_preds"], results["test_labels"],
                           cliff_mask, label_stats)
        if rc is not None:
            results["test"]["rmse_cliff"] = rc
            results["test"]["n_cliff"] = n
    if task_type == "regression":
        results["test_preds"] = to_original_scale(results["test_preds"], label_stats)
        results["test_labels"] = to_original_scale(results["test_labels"], label_stats)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def make_ctx(task_type, num_tasks, template, device, logger):
    return SimpleNamespace(task_type=task_type, num_tasks=num_tasks,
                           template=template, device=device, logger=logger)
