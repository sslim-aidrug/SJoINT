"""SJoINT contrastive pre-training.
    python pretrain/train.py --data_path data/processed/zinc250k --vocab_path data/vocab/vocab_zinc.json --checkpoint_dir checkpoints/backbone
"""
import os
import sys
import glob
import json
import time
import math
import random
import logging
import argparse

import numpy as np
import torch
import torch.multiprocessing as _torch_mp
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

try:
    _torch_mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from model import SJoINTModel
from utils import GraphPairDataset, custom_collate, nt_xent_loss

_MASK_KEYS = ('jt2mol_b', 'jt2mol_r', 'jt2mol_c',
              'mol2jt_b', 'mol2jt_r', 'mol2jt_c')


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def setup_logger(checkpoint_dir, resume=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh = logging.FileHandler(
        os.path.join(checkpoint_dir, "train.log"),
        mode='a' if resume else 'w')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch <= self.warmup_epochs:
            scale = epoch / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def _to_device(batch, device):
    jt_b = batch['jt_batch'].to(device, non_blocking=True)
    mol_b = batch['mol_batch'].to(device, non_blocking=True)
    masks = tuple(batch[k].to(device, non_blocking=True) for k in _MASK_KEYS)
    return jt_b, mol_b, masks


def train_one_epoch(model, loader, optimizer, scaler, device, temperature, epoch,
                    amp_dtype=torch.float16):
    model.train()
    total_loss = 0.0
    use_scaler = scaler is not None and scaler.is_enabled()
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False, dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
        jt_batch, mol_batch, masks = _to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
            p_jt, p_mol = model(jt_batch, mol_batch, *masks)
        loss = nt_xent_loss(p_jt.float(), p_mol.float(), temperature=temperature)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        cur = loss.item()
        total_loss += cur
        pbar.set_postfix(loss=f"{cur:.4f}", avg=f"{total_loss/(batch_idx+1):.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, temperature, amp_dtype=torch.float16):
    model.eval()
    total_loss = torch.zeros((), device=device)
    n = 0
    for batch in tqdm(loader, desc='[Eval]', leave=False, dynamic_ncols=True):
        if batch is None:
            continue
        jt_batch, mol_batch, masks = _to_device(batch, device)
        with autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
            p_jt, p_mol = model(jt_batch, mol_batch, *masks)
        loss = nt_xent_loss(p_jt.float(), p_mol.float(), temperature=temperature)
        total_loss += loss.detach()
        n += 1
    return (total_loss / max(n, 1)).item()


def _load_vocab_size(vocab_path):
    with open(vocab_path) as f:
        return len(json.load(f)["vocab"])


def _infer_jt_feature_dim(data_path):
    chunk_paths = sorted(glob.glob(os.path.join(data_path, "*.pt")))
    if not chunk_paths:
        raise FileNotFoundError(f"No .pt files in {data_path}")
    sample = torch.load(chunk_paths[0], map_location="cpu", weights_only=False)
    if isinstance(sample, list):
        sample = sample[0]
    return int(sample["jt_features"].shape[-1])


def _save_checkpoint(path, model, optimizer, scaler, epoch, best_loss, best_epoch):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
    }, path)


def main(args):
    jt_vocab_size = _load_vocab_size(args.vocab_path)
    jt_feature_dim = _infer_jt_feature_dim(args.data_path)

    resume = args.resume_ckpt and os.path.exists(args.resume_ckpt)
    logger = setup_logger(args.checkpoint_dir, resume=resume)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    num_chunks = len(glob.glob(os.path.join(args.data_path, "*.pt")))
    chunk_perm = torch.randperm(num_chunks, generator=torch.Generator().manual_seed(args.seed)).tolist()
    split = int(0.8 * num_chunks)
    train_idx, val_idx = sorted(chunk_perm[:split]), sorted(chunk_perm[split:])

    train_ds = GraphPairDataset(args.data_path, split_indices=train_idx)
    val_ds = GraphPairDataset(args.data_path, split_indices=val_idx)

    loader_kw = dict(
        batch_size=args.batch_size, collate_fn=custom_collate,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        worker_init_fn=_seed_worker)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, generator=g, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = SJoINTModel(
        jt_vocab_size=jt_vocab_size,
        jt_feature_dim=jt_feature_dim,
        dropout=args.dropout, ca_dropout=args.ca_dropout,
    ).to(device)

    if args.compile:
        try:
            model = torch.compile(model, mode="default")
            logging.info("  torch.compile enabled (mode=default)")
        except Exception as e:
            logging.warning(f"torch.compile failed, falling back to eager: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(torch.cuda.is_available() and amp_dtype == torch.float16))

    start_epoch = 1
    best_loss, best_epoch = float('inf'), 0
    if resume:
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scaler.load_state_dict(ckpt['scaler'])
            start_epoch = ckpt['epoch'] + 1
            best_loss = ckpt['best_loss']
            best_epoch = ckpt['best_epoch']
        else:
            model.load_state_dict(ckpt)
        logger.info(f"  Resumed from {args.resume_ckpt} (epoch {start_epoch}, best={best_loss:.6f})")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("=" * 65)
    logger.info("  SJoINT Pretrain")
    logger.info("=" * 65)
    logger.info(f"  Device : {device}  |  Params : {n_params:,}")
    logger.info(f"  Train  : {len(train_ds):,} ({len(train_idx)} chunks)")
    logger.info(f"  Val    : {len(val_ds):,} ({len(val_idx)} chunks)")
    logger.info(f"  Vocab  : {jt_vocab_size}")
    logger.info(f"  JT Feat: {jt_feature_dim}")
    logger.info(f"  LR: {args.learning_rate}  WD: {args.weight_decay}  "
                f"Temp: {args.temperature}  Warmup: {args.warmup_epochs}  "
                f"Dropout: gnn {args.dropout} / ca {args.ca_dropout}")
    logger.info(f"  CA Blocks: {len(model.blocks)}")
    logger.info(f"  BS: {args.batch_size}  Epochs: {start_epoch}-{args.max_epochs}")
    logger.info(f"  AMP dtype: {amp_dtype}  scaler: {scaler.is_enabled()}")
    logger.info("=" * 65)

    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.max_epochs)

    for epoch in range(1, args.max_epochs + 1):
        scheduler.step(epoch)
        if epoch < start_epoch:
            continue
        start = time.time()
        t_loss = train_one_epoch(model, train_loader, optimizer, scaler, device,
                                 args.temperature, epoch, amp_dtype=amp_dtype)
        v_loss = evaluate(model, val_loader, device, args.temperature,
                          amp_dtype=amp_dtype)

        logger.info(
            f"Epoch {epoch:03d} | Train: {t_loss:.6f} | Val: {v_loss:.6f} | "
            f"LR: {scheduler.get_lr():.6f} | {time.time()-start:.1f}s")

        if v_loss < best_loss:
            best_loss, best_epoch = v_loss, epoch
            _save_checkpoint(
                os.path.join(args.checkpoint_dir, "best_model.ckpt"),
                model, optimizer, scaler, epoch, best_loss, best_epoch)
            logger.info(f"  --> Best (Loss: {best_loss:.6f})")

        _save_checkpoint(
            os.path.join(args.checkpoint_dir, "last_model.ckpt"),
            model, optimizer, scaler, epoch, best_loss, best_epoch)

    logger.info(f"Done. Best: {best_loss:.4f} (epoch {best_epoch})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SJoINT contrastive pre-training")
    parser.add_argument('--data_path', type=str, default="./data/processed/zinc250k/",
                        help="directory of .pt chunks")
    parser.add_argument('--vocab_path', type=str, default="./data/vocab/vocab_zinc.json",
                        help="JT vocabulary")
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints/")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.0, help="GNN encoder dropout")
    parser.add_argument('--ca_dropout', type=float, default=0.1, help="cross-attention dropout")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--compile', action='store_true', help="torch.compile the model")
    parser.add_argument('--bf16', action='store_true', default=True, help="bfloat16 AMP (default)")
    parser.add_argument('--no-bf16', dest='bf16', action='store_false', help="fp16 AMP with GradScaler")
    parser.add_argument('--resume_ckpt', type=str, default='', help="checkpoint to resume from")

    main(parser.parse_args())
