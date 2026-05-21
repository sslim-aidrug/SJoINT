# SJoINT
**Substructure-Driven Junction Tree for Interpretable ADMET Prediction**

Code accompanying the SJoINT paper: pre-training on ZINC250K and fine-tuning
on MoleculeNet (random / scaffold splits) and MoleculeACE.

## Requirements

- Python 3.11+
- PyTorch 2.7+ (CUDA 12.4+ recommended)
- PyTorch Geometric 2.7+
- RDKit, NumPy, SciPy, scikit-learn, tqdm
- NVIDIA GPU (tested on B200, sm_100)

## Installation

```bash
mamba env create -f environment.yml
mamba activate sj
```

Pip alternative: install a CUDA-matched PyTorch + PyG wheel manually, then
`pip install -r requirements.txt`.

## Data

A toy subset (500 ZINC SMILES + FreeSolv 642 mols) ships under
[`data/toy/`](data/toy/README.md) for smoke-testing the pipeline.
Full corpora are not shipped — place them under `data/` and see
[`preprocess/README.md`](preprocess/README.md) for the expected layout.

| Corpus | Source |
|---|---|
| ZINC250K (pre-training) | https://github.com/wengong-jin/icml18-jtnn |
| MoleculeNet (fine-tuning) | https://moleculenet.org/datasets-1 |
| MoleculeACE (cliff benchmark) | https://github.com/molML/MoleculeACE |

## Running SJoINT

### 1. Preprocessing

```bash
# JT vocabulary + ZINC tensorization
python -m preprocess.pretrain.build_vocab \
    --input data/ZINC250K/jtnn_zinc250k.txt \
    --output data/ZINC250K/vocab.json --workers 60
python -m preprocess.pretrain.tensorize \
    --input data/ZINC250K/jtnn_zinc250k.txt \
    --output data/ZINC250K/data \
    --vocab data/ZINC250K/vocab.json \
    --workers 60 --chunk_size 50000

# MoleculeNet split + tensorize (random + scaffold, 10 seeds)
python -m preprocess.moleculenet.split \
    --mode both --csv_dir data/MoleculeNet/raw \
    --out_dir data/MoleculeNet/split \
    --seeds 1 2 3 42 43 44 123 456 789 1024
python -m preprocess.moleculenet.tensorize \
    --data_dir data/MoleculeNet/split \
    --vocab data/ZINC250K/vocab.json \
    --modes random scaffold \
    --seeds 1 2 3 42 43 44 123 456 789 1024

# MoleculeACE
python -m preprocess.moleculeace.tensorize \
    --csv_dir data/MoleculeACE/raw \
    --out_dir data/MoleculeACE/processed \
    --vocab data/ZINC250K/vocab.json --workers 30
```

### 2. Pre-training

```bash
python pretrain/train.py \
    --data_path data/ZINC250K \
    --checkpoint_dir pretrain/checkpoints/sjoint \
    --learning_rate 5e-4 --weight_decay 5e-5 --batch_size 512 \
    --max_epochs 100 --gpu_id 0
```

### 3. Fine-tuning

```bash
# MoleculeNet — random split (uses BEST_HP from finetune/configs.py)
python finetune/run.py \
    --dataset BBBP --seed 1 \
    --pretrain_ckpt pretrain/checkpoints/sjoint/best_model.ckpt --gpu_id 0

# MoleculeNet — scaffold split
python finetune/train.py \
    --dataset BBBP --seed 1 \
    --pretrain_ckpt pretrain/checkpoints/sjoint/best_model.ckpt --gpu_id 0 \
    --data_base data/MoleculeNet/split/scaffold

# MoleculeACE
python finetune/run_moleculeace.py --datasets CHEMBL204_Ki CHEMBL234_Ki
```

## Project Structure

```
SJoINT/
├── preprocess/        SMILES → .pt tensors
│   ├── core/            Featurization + utilities
│   ├── pretrain/        JT vocabulary + ZINC tensorization
│   ├── moleculenet/     Random / scaffold split + tensorization
│   └── moleculeace/     Canonical split tensorization
├── pretrain/          Contrastive pre-training (model + train loop + HP sweep)
└── finetune/          Two-stage fine-tuning (model + train loop + benchmark runners)
```
