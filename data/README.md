# data/

Raw and processed data live here. The small **raw inputs** (SMILES, MoleculeNet
CSVs) and the **JT vocabulary** are shipped; the large **processed tensors** are
built locally and not tracked by git (see `.gitignore`).

## Expected layout

```
data/
├── raw/
│   ├── zinc250k.txt                 # pre-training SMILES, one per line
│   └── moleculenet/                 # downstream CSVs (BBBP, BACE, ...)
├── vocab/
│   └── vocab_zinc.json              # JT substructure vocabulary (built by preprocess/pretrain/build_vocab.py)
└── processed/
    ├── zinc250k/                    # pretrain .pt chunks + vocab.json  (preprocess/pretrain/tensorize.py)
    └── moleculenet/                 # per-dataset train/val/test .pt     (preprocess/moleculenet/tensorize.py)
```

## How to obtain / build

1. Put raw SMILES / CSVs under `data/raw/`.
2. Build the JT vocabulary — see `preprocess/pretrain/build_vocab.py`.
3. Tensorize — see `preprocess/README.md` (`preprocess/run.sh`).

The exact paths are passed as CLI args, so any layout works; the above is the
convention the READMEs assume.
