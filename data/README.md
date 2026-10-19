# data/

The **raw inputs** (ZINC250K SMILES, MoleculeNet and MoleculeACE CSVs) and the
**JT vocabulary** of the released backbone are shipped; the large **processed
tensors** are built locally and not tracked by git (see `.gitignore`).

## Expected layout

```
data/
├── raw/
│   ├── zinc250k.txt                 # pre-training SMILES, one per line
│   ├── moleculenet/                 # downstream CSVs (BBBP, BACE, ...)
│   └── moleculeace/<task>/          # train.csv, test.csv (fixed MoleculeACE splits)
├── vocab/
│   └── vocab_zinc.json              # JT vocabulary of the released backbone
└── processed/
    ├── zinc250k/                    # pre-training .pt chunks          (preprocess/pretrain/tensorize.py)
    ├── moleculenet/                 # per-dataset train/val/test .pt   (preprocess/moleculenet/tensorize.py)
    └── moleculeace/                 # per-task train/test .pt + cliff_mask.pt (preprocess/moleculeace/tensorize.py)
```

## How to build

Tensorize with `preprocess/run.sh` (see `preprocess/README.md`). Use the shipped
`vocab/vocab_zinc.json` with the released backbone; a new vocabulary built with
`preprocess/pretrain/build_vocab.py` requires pre-training a new backbone.

The exact paths are passed as CLI args, so any layout works; the above is the
convention the READMEs assume.
