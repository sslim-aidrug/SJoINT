# data/

Raw inputs and the JT vocabulary are shipped; processed tensors are built with `preprocess/run.sh` and git-ignored.

```
data/
├── raw/
│   ├── zinc250k.txt                 pre-training SMILES, one per line
│   ├── moleculenet/*.csv            MoleculeNet datasets
│   └── moleculeace/<task>/          train.csv, test.csv (fixed MoleculeACE splits)
├── vocab/vocab_zinc.json            JT vocabulary of the released backbone
└── processed/
    ├── zinc250k/                    pre-training .pt chunks
    ├── moleculenet/                 per-dataset train/val/test .pt
    └── moleculeace/                 per-task train/test .pt + cliff_mask.pt
```
