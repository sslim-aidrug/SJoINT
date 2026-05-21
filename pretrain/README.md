# Pretrain

SimCLR contrastive pre-training for Molecular Graph + Junction Tree joint representations.

## Architecture

```
MolEncoder (GATv2 + JK, 3-layer) ──┐
                                    ├─► Cross-Attention Blocks (×3) ──► LayerNorm ──► MeanPool ──► Projection
JTEncoder  (GATv2, 2-layer) ───────┘
```

## Files

| File | Description |
|---|---|
| `model.py` | `SJoINTModel` — encoders, cross-attention, projection head |
| `train.py` | Training loop with AMP, cosine warmup, checkpointing |
| `utils.py` | `GraphPairDataset`, collate function, SimCLR loss |
| `run_pretrain_sweep.py` | Multi-GPU HP sweep launcher |
| `run.sh` | Unified entry script (`single` / `sweep`) |

## Quick Start

```bash
# Single training job
bash pretrain/run.sh single

# HP sweep across GPU 0 and 1, 10 processes per GPU
bash pretrain/run.sh sweep

# Dry run (print combos without launching)
bash pretrain/run.sh sweep --dry_run
```

## HP Sweep Grid

24 combinations:

| Parameter | Values |
|---|---|
| `learning_rate` | `5e-5`, `1e-4`, `2e-4`, `5e-4` |
| `weight_decay` | `1e-5`, `5e-5`, `1e-4` |
| `batch_size` | `256`, `512` |

Default execution: 2 GPUs × 10 processes = 20 concurrent jobs → 24 combos run in ~2 waves.

Per-config output: `sweep_results/lr<LR>_wd<WD>_bs<BS>/{best_model.ckpt, last_model.ckpt, train.log}`
Per-config stdout/stderr: `sweep_results/_logs/<tag>.log`

## Single Run Arguments (`bash pretrain/run.sh single`)

| Variable | Default | Description |
|---|---|---|
| `GPU_ID` | `0` | CUDA device |
| `LR` | `5e-4` | Learning rate |
| `WD` | `5e-5` | Weight decay |
| `BS` | `512` | Batch size |
| `EPOCHS` | `100` | Max epochs |
| `OUT` | `pretrain/checkpoints/lr<LR>_wd<WD>_bs<BS>` | Checkpoint dir |

## Sweep Variables (`bash pretrain/run.sh sweep`)

| Variable | Default | Description |
|---|---|---|
| `GPUS` | `0,1` | Comma-separated GPU ids |
| `PROCS_PER_GPU` | `10` | Concurrent training procs per GPU |
| `RESULTS_DIR` | `pretrain/sweep_results` | Output dir |
| `DATA_PATH` | `data/ZINC250K` | ZINC250K data dir (`.pt` chunks + `vocab.json`) |

`SJOINT_PYTHON` and `SJOINT_DATA` env vars (or `--gpus`/`--procs_per_gpu`/`--results_dir`
CLI flags) are also accepted by `run_pretrain_sweep.py` directly.

## Direct `train.py` Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `./data/zinc250k/` | Directory with `.pt` chunks and `vocab.json` |
| `--checkpoint_dir` | `./checkpoints/` | Output directory for checkpoints and logs |
| `--learning_rate` | `1e-4` | Learning rate |
| `--temperature` | `0.07` | SimCLR temperature |
| `--weight_decay` | `1e-5` | AdamW weight decay |
| `--warmup_epochs` | `10` | Linear warmup epochs |
| `--num_ca_blocks` | `3` | Number of cross-attention blocks |
| `--proj_dim` | `32` | Projection head output dim |
| `--batch_size` | `512` | Batch size |
| `--max_epochs` | `100` | Maximum training epochs |
| `--seed` | `42` | Random seed |
| `--gpu_id` | `0` | CUDA device ID |
| `--num_workers` | `8` | DataLoader workers |
| `--resume_ckpt` | | Path to checkpoint for resuming |

## Fixed Hyperparameters (in model)

| Parameter | Value |
|---|---|
| `hidden_dim` | 32 |
| Encoder norm | BatchNorm |
| CA norm | LayerNorm (pre-norm) |
| Final norm | LayerNorm |
| Pooling | Mean |
| `enc_dropout` | 0.1 |
| `ca_dropout` | 0.2 |

## Output Layout

```
pretrain/
├── checkpoints/                       # Single runs
│   └── <tag>/
│       ├── best_model.ckpt
│       ├── last_model.ckpt
│       └── train.log
└── sweep_results/                     # HP sweep
    ├── _logs/                         # Per-config stdout/stderr
    │   └── <tag>.log
    └── <tag>/                         # Per-config training output
        ├── best_model.ckpt
        ├── last_model.ckpt
        └── train.log
```

`<tag>` format: `lr<LR>_wd<WD>_bs<BS>`, e.g. `lr0.0005_wd5e-05_bs512`.
