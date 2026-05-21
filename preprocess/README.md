# Preprocessing

Pipeline for converting raw SMILES data into `.pt` tensors for model training.
Three benchmarks are handled by three sibling sub-packages.

## Structure

```
preprocess/
├── run.sh                          Unified entry script
├── core/                           Shared modules
│   ├── configs.py                  MoleculeNet dataset metadata (10 datasets)
│   ├── features.py                 Chemistry featuriser (atom 51D, bond 12D, JT 10D)
│   └── utils.py                    Vocab loading, tensor conversion, chunk I/O
├── pretrain/                       Pre-training (ZINC250K)
│   ├── build_vocab.py              JT substructure vocabulary builder
│   └── tensorize.py                SMILES → .pt chunks
├── moleculenet/                    MoleculeNet fine-tune benchmark
│   ├── split.py                    CSV → train/val/test split (random / scaffold / both)
│   └── tensorize.py                Split CSV → .pt
└── moleculeace/                    MoleculeACE benchmark (canonical split)
    └── tensorize.py                Pre-split CSV → .pt + cliff_metadata.json
```

## Pipeline

```
Pretrain     :  ZINC SMILES ──→ build_vocab ──→ vocab.json
                          └──→ tensorize  ──→ data_chunk_*.pt

MoleculeNet  :  raw/*.csv ──→ split ──→ seed{n}/*.csv ──→ tensorize ──→ *.pt
                              (random / scaffold / both, ≥1 seed)

MoleculeACE  :  pre-split CSV (split column) ──→ tensorize ──→ {train,val,test}.pt
                + cliff_metadata.json (cliff_mols, smiles, y for test)
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
# (writes to data/MoleculeNet/split/{random,scaffold}/...)
MODE=both bash preprocess/run.sh moleculenet

# Custom seed list
SEEDS="42 43 44" bash preprocess/run.sh moleculenet

# 3) MoleculeACE (canonical train/test split from van Tilborg 2022)
bash preprocess/run.sh moleculeace
```

### Individual Commands

```bash
# JT Vocabulary
python -m preprocess.pretrain.build_vocab \
    --input data/ZINC250K/jtnn_zinc250k.txt \
    --output data/ZINC250K/vocab.json \
    --workers 60

# Pre-train tensorize
python -m preprocess.pretrain.tensorize \
    --input data/ZINC250K/jtnn_zinc250k.txt \
    --output data/ZINC250K/data \
    --vocab data/ZINC250K/vocab.json \
    --workers 60 --chunk_size 50000

# MoleculeNet split (10 seeds, random)
python -m preprocess.moleculenet.split \
    --mode random \
    --csv_dir data/MoleculeNet/raw \
    --out_dir data/MoleculeNet/random_split \
    --seeds 1 2 3 42 43 44 123 456 789 1024

# Both random and scaffold at once (writes to <out_dir>/random and <out_dir>/scaffold)
python -m preprocess.moleculenet.split \
    --mode both \
    --csv_dir data/MoleculeNet/raw \
    --out_dir data/MoleculeNet/split \
    --seeds 1 2 3 42 43 44 123 456 789 1024

# MoleculeNet tensorize (single mode)
python -m preprocess.moleculenet.tensorize \
    --data_dir data/MoleculeNet/random_split \
    --vocab data/ZINC250K/vocab.json \
    --seeds 1 2 3 42 43 44 123 456 789 1024

# MoleculeNet tensorize (both modes; data_dir is parent of random/ and scaffold/)
python -m preprocess.moleculenet.tensorize \
    --data_dir data/MoleculeNet/split \
    --vocab data/ZINC250K/vocab.json \
    --modes random scaffold \
    --seeds 1 2 3 42 43 44 123 456 789 1024

# Auto-discover seed* dirs (handy if seed list isn't remembered)
python -m preprocess.moleculenet.tensorize \
    --data_dir data/MoleculeNet/random_split \
    --vocab data/ZINC250K/vocab.json \
    --auto

# MoleculeACE tensorize (one CSV per ChEMBL task, with `split` column)
python -m preprocess.moleculeace.tensorize \
    --csv_dir data/MoleculeACE/raw \
    --out_dir data/MoleculeACE/processed \
    --vocab data/ZINC250K/vocab.json \
    --workers 30
```

## Environment Variables (run.sh)

| Variable | Default | Description |
|---|---|---|
| `PYTHON` | `python` | Python binary |
| `MODE` | `random` | Split mode (`random` / `scaffold` / `both`) — `finetune` only |
| `CSV_DIR` | `data/MoleculeNet/raw` (moleculenet) / `data/MoleculeACE/raw` (moleculeace) | Raw CSV directory |
| `OUT_DIR` | depends on `MODE` (finetune) / `data/MoleculeACE/processed` (moleculeace) | Output directory |
| `SEEDS` | `1 2 3 42 43 44 123 456 789 1024` | Space-separated seed list (finetune) |
| `WORKERS` | `60` (pretrain) / `30` (moleculeace) | Parallel workers |
| `CHUNK_SIZE` | `50000` | Pre-train chunk size |

## Expected Data Layout

```
data/
├── ZINC250K/                         Pre-train corpus
│   ├── jtnn_zinc250k.txt             SMILES (one per line)
│   ├── vocab.json                    JT substructure vocabulary
│   └── data_chunk_{0..N}.pt          Tensorised chunks
├── MoleculeNet/                      MoleculeNet fine-tune benchmarks
│   ├── raw/                          Original CSVs
│   │   ├── BACE.csv  BBBP.csv  ClinTox.csv  ESOL.csv  FreeSolv.csv
│   │   └── HIV.csv  Lipophilicity.csv  SIDER.csv  Tox21.csv  ToxCast.csv
│   ├── random_split/                 Random split — N seeds × train/val/test
│   │   └── {dataset}/seed{n}/        .csv + .pt
│   ├── scaffold_split/               Scaffold split — N seeds × train/val/test
│   │   └── {dataset}/seed{n}/        .csv + .pt
│   └── split/                        (MODE=both) holds both random/ and scaffold/
│       └── {random,scaffold}/{dataset}/seed{n}/
└── MoleculeACE/
    ├── raw/                                    30 canonical pre-split CSVs (CHEMBL*_*)
    └── processed/                              Tensorised output
        ├── metadata.json
        └── {CHEMBL_TASK}/
            ├── train.pt
            ├── val.pt                          first 10 % of canonical train
            ├── test.pt                         canonical test
            └── cliff_metadata.json             cliff_mols, smiles, y for test
```

The raw CSVs in `data/MoleculeACE/raw/` are the canonical pre-split files released by
the MoleculeACE benchmark (van Tilborg et al., 2022), available at
<https://github.com/molML/MoleculeACE> under `MoleculeACE/Data/benchmark_data/old/`.
Clone that repository and copy the CSVs into place, e.g.

```bash
git clone https://github.com/molML/MoleculeACE.git /tmp/MoleculeACE
mkdir -p data/MoleculeACE/raw
cp -u /tmp/MoleculeACE/MoleculeACE/Data/benchmark_data/old/*.csv data/MoleculeACE/raw/
```
