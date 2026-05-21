"""SJoINT contrastive pretraining loop."""
import os
import sys
import glob
import json
import time
import math
import logging
import argparse

import numpy as np
import torch
import torch.multiprocessing as _torch_mp
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Avoid SCM_RIGHTS fd-passing exhaustion under many concurrent DataLoader procs
# (failure mode: "received 0 items of ancdata" -> Pin memory thread exits).
try:
    _torch_mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from model import SJoINTModel
from utils import GraphPairDataset, custom_collate, simclr_loss

_MASK_KEYS = ('jt2mol_b', 'jt2mol_r', 'jt2mol_c',
              'mol2jt_b', 'mol2jt_r', 'mol2jt_c')


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
            loss = simclr_loss(p_jt, p_mol, temperature=temperature)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # bfloat16 has fp32-equivalent exponent range — no loss-scaling needed.
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
            loss = simclr_loss(p_jt, p_mol, temperature=temperature)
        total_loss += loss.detach()
        n += 1
    return (total_loss / max(n, 1)).item()


def _load_vocab_size(data_path):
    vocab_path = os.path.join(data_path, "vocab.json")
    if not os.path.exists(vocab_path):
        vocab_path = os.path.join(os.path.dirname(data_path.rstrip("/")), "vocab.json")
    with open(vocab_path) as f:
        return len(json.load(f)["vocab"])


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
    jt_vocab_size = _load_vocab_size(args.data_path)

    resume = args.resume_ckpt and os.path.exists(args.resume_ckpt)
    logger = setup_logger(args.checkpoint_dir, resume=resume)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # B200 (sm_100) - prefer high-precision TF32/BF16 matmul path
        torch.set_float32_matmul_precision("high")

    # AMP dtype: bfloat16 default on Blackwell (B200) — same exponent range as fp32,
    # no GradScaler needed and avoids the fp16 underflow path.
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    num_chunks = len(glob.glob(os.path.join(args.data_path, "*.pt")))
    chunk_perm = torch.randperm(num_chunks, generator=torch.Generator().manual_seed(args.seed)).tolist()
    split = int(0.8 * num_chunks)
    train_idx, test_idx = sorted(chunk_perm[:split]), sorted(chunk_perm[split:])

    train_ds = GraphPairDataset(args.data_path, split_indices=train_idx)
    test_ds = GraphPairDataset(args.data_path, split_indices=test_idx)

    loader_kw = dict(
        batch_size=args.batch_size, collate_fn=custom_collate,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    model = SJoINTModel(
        jt_vocab_size=jt_vocab_size,
        num_ca_blocks=args.num_ca_blocks,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        mol_num_layers=args.mol_num_layers,
        jt_num_layers=args.jt_num_layers,
        dropout=args.dropout,
        ca_dropout=args.ca_dropout,
    ).to(device)

    if args.compile:
        # mode="default" is most permissive for dynamic shapes (PyG graphs vary per batch).
        # "reduce-overhead" wants static shapes via CUDA graphs and tends to recompile a lot.
        try:
            model = torch.compile(model, mode="default")
            logging.info("  torch.compile enabled (mode=default)")
        except Exception as e:
            logging.warning(f"torch.compile failed, falling back to eager: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # GradScaler only for fp16; bf16 has fp32 dynamic range.
    scaler = GradScaler(enabled=(torch.cuda.is_available() and amp_dtype == torch.float16))

    # Resume from checkpoint
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
            # Legacy: plain state_dict
            model.load_state_dict(ckpt)
            start_epoch = args.start_epoch
        logger.info(f"  Resumed from {args.resume_ckpt} (epoch {start_epoch}, best={best_loss:.6f})")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("=" * 65)
    logger.info("  SJoINT Pretrain")
    logger.info("=" * 65)
    logger.info(f"  Device : {device}  |  Params : {n_params:,}")
    logger.info(f"  Train  : {len(train_ds):,} ({len(train_idx)} chunks)")
    logger.info(f"  Test   : {len(test_ds):,} ({len(test_idx)} chunks)")
    logger.info(f"  Vocab  : {jt_vocab_size}")
    logger.info(f"  LR: {args.learning_rate}  WD: {args.weight_decay}  "
                f"Temp: {args.temperature}  Warmup: {args.warmup_epochs}")
    logger.info(f"  CA Blocks: {args.num_ca_blocks}")
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
        v_loss = evaluate(model, test_loader, device, args.temperature,
                          amp_dtype=amp_dtype)

        logger.info(
            f"Epoch {epoch:03d} | Train: {t_loss:.6f} | Test: {v_loss:.6f} | "
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
    parser.add_argument('--data_path', type=str, default="./data/zinc250k/",
                        help="Directory containing .pt chunk files and vocab.json")
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints/")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--num_ca_blocks', type=int, default=3)
    parser.add_argument('--proj_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--mol_num_layers', type=int, default=3)
    parser.add_argument('--jt_num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--ca_dropout', type=float, default=0.2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ffn_ratio', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--compile', action='store_true',
                        help="Wrap model with torch.compile (Inductor)")
    parser.add_argument('--bf16', action='store_true', default=True,
                        help="Use bfloat16 AMP (default True on B200). "
                             "Pass --no-bf16 to fall back to fp16+GradScaler.")
    parser.add_argument('--no-bf16', dest='bf16', action='store_false')
    parser.add_argument('--resume_ckpt', type=str, default='',
                        help="Path to checkpoint for resuming training")
    parser.add_argument('--start_epoch', type=int, default=1,
                        help="Start epoch (only used with legacy checkpoints)")

    main(parser.parse_args())
