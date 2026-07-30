# Preprocessing

Pipeline for converting raw SMILES data into `.pt` tensors for model training.
Three benchmarks are handled by three sibling sub-packages.

## Structure

```
preprocess/
├── run.sh                          Unified entry script
├── core/                           Shared modules
│   ├── configs.py                  MoleculeNet dataset metadata
│   ├── features.py                 Chemistry featuriser (atom 51D, bond 12D, JT 10D)
│   └── utils.py                    Vocab loading, tensor conversion, chunk I/O
├── pretrain/                       Pre-training (ZINC250K)
│   ├── build_vocab.py              JT substructure vocabulary builder
│   └── tensorize.py                SMILES → .pt chunks
├── moleculenet/                    MoleculeNet fine-tune benchmark
│   ├── split.py                    CSV → train/val/test split (random / scaffold / both)
│   └── tensorize.py                Split CSV → .pt
└── moleculeace/                    MoleculeACE activity-cliff benchmark
    └── tensorize.py                Pre-split CSV → .pt + cliff_mask
```

JT node features are `jt_features_10d ⊕ atom_feature_mean_51d` (**61-D**); this
must match the backbone's `jt_feature_dim` (the released backbone is 61-D).

## Pipeline

```
Pretrain     :  ZINC SMILES ──→ build_vocab ──→ vocab_zinc.json
                          └──→ tensorize  ──→ data_chunk_*.pt

MoleculeNet  :  raw/*.csv ──→ split ──→ seed{n}/*.csv ──→ tensorize ──→ *.pt
                              (random / scaffold / both, ≥1 seed)

MoleculeACE  :  raw/<task>/{train,test}.csv ──→ tensorize ──→ *.pt + cliff_mask.pt
                              (single fixed train/test split per task)
```

## Quick Start

```bash
# 1) Pre-training (vocab + tensorize)
bash preprocess/run.sh pretrain

# 2) MoleculeNet (random split + tensorize, default 10 seeds)
bash preprocess/run.sh moleculenet
# (`finetune` is kept as a backward-compatible alias)

# Scaffold split
MODE=scaffold bash preprocess/run.sh moleculenet

# Both random and scaffold in one run
# (writes to data/processed/moleculenet/split/{random,scaffold}/...)
MODE=both bash preprocess/run.sh moleculenet

# Custom seed list
SEEDS="42 43 44" bash preprocess/run.sh moleculenet

# 3) MoleculeACE (tensorize pre-split activity-cliff CSVs)
bash preprocess/run.sh moleculeace
```

### Individual Commands

```bash
# JT Vocabulary
python -m preprocess.pretrain.build_vocab \
    --input data/raw/zinc250k.txt \
    --output data/vocab/vocab_zinc.json \
    --workers 60

# Pre-train tensorize
python -m preprocess.pretrain.tensorize \
    --input data/raw/zinc250k.txt \
    --output data/processed/zinc250k/data \
    --vocab data/vocab/vocab_zinc.json \
    --workers 60 --chunk_size 50000

# MoleculeNet split (10 seeds, random)
python -m preprocess.moleculenet.split \
    --mode random \
    --csv_dir data/raw/moleculenet \
    --out_dir data/processed/moleculenet/random_split \
    --seeds 1 2 3 42 43 44 123 456 789 1024

# Both random and scaffold at once (writes to <out_dir>/random and <out_dir>/scaffold)
python -m preprocess.moleculenet.split \
    --mode both \
    --csv_dir data/raw/moleculenet \
    --out_dir data/processed/moleculenet/split \
    --seeds 1 2 3 42 43 44 123 456 789 1024

# MoleculeNet tensorize (single mode)
python -m preprocess.moleculenet.tensorize \
    --data_dir data/processed/moleculenet/random_split \
    --vocab data/vocab/vocab_zinc.json \
    --seeds 1 2 3 42 43 44 123 456 789 1024

# MoleculeNet tensorize (both modes; data_dir is parent of random/ and scaffold/)
python -m preprocess.moleculenet.tensorize \
    --data_dir data/processed/moleculenet/split \
    --vocab data/vocab/vocab_zinc.json \
    --modes random scaffold \
    --seeds 1 2 3 42 43 44 123 456 789 1024

# Auto-discover seed* dirs (handy if seed list isn't remembered)
python -m preprocess.moleculenet.tensorize \
    --data_dir data/processed/moleculenet/random_split \
    --vocab data/vocab/vocab_zinc.json \
    --auto

# MoleculeACE tensorize (pre-split CSV → .pt + cliff_mask)
python -m preprocess.moleculeace.tensorize \
    --data_dir data/raw/moleculeace \
    --out_dir data/processed/moleculeace \
    --vocab data/vocab/vocab_zinc.json \
    --workers 30
```

## Environment Variables (run.sh)

| Variable | Default | Description |
|---|---|---|
| `PYTHON` | `python` | Python binary |
| `MODE` | `random` | Split mode (`random` / `scaffold` / `both`) — moleculenet only |
| `CSV_DIR` | `data/raw/moleculenet` | Raw CSV directory |
| `OUT_DIR` | depends on `MODE` | Output directory |
| `SEEDS` | `1 2 3 42 43 44 123 456 789 1024` | Space-separated seed list (moleculenet only) |
| `WORKERS` | `60` | Parallel workers |
| `CHUNK_SIZE` | `50000` | Pre-train chunk size |

## Expected Data Layout

```
data/
├── raw/
│   ├── zinc250k.txt                  Pre-train SMILES (one per line)
│   ├── moleculenet/                  Original CSVs
│   │   ├── BACE.csv  BBBP.csv  ClinTox.csv  ESOL.csv  FreeSolv.csv
│   │   └── Lipophilicity.csv  SIDER.csv  Tox21.csv  ToxCast.csv
│   └── moleculeace/                  30 ChEMBL activity-cliff tasks
│       └── <task>/{train.csv, test.csv}   (test.csv carries a cliff_mol flag)
├── vocab/
│   └── vocab_zinc.json               JT substructure vocabulary
└── processed/                        Tensorised outputs (git-ignored)
    ├── zinc250k/
    │   └── data_chunk_{0..N}.pt      Pre-train chunks
    ├── moleculenet/
    │   ├── random_split/{dataset}/seed{n}/     .csv + .pt
    │   ├── scaffold_split/{dataset}/seed{n}/   .csv + .pt
    │   └── split/{random,scaffold}/{dataset}/seed{n}/   (MODE=both)
    └── moleculeace/{task}/               train.pt + test.pt + cliff_mask.pt + metadata.json
```

MoleculeACE = 30 ChEMBL activity-cliff tasks (van Tilborg 2022), regression,
metric **RMSE + RMSE_cliff**. Following the original protocol, each task has a
**single fixed train/test split** (`train.csv` / `test.csv`, the latter carrying a
per-molecule `cliff_mol` flag). Hyper-parameters are chosen by k-fold
cross-validation on the training set and the final model is retrained on the full
training set — see `finetune/moleculeace/`.
