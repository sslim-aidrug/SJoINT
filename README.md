# SJoINT

**S**ubstructure-Driven **J**uncti**o**n Tree for **INT**erpretable Molecular Property Prediction.

SJoINT encodes each molecule as an atom-level molecular graph and a substructure-level junction tree (JT),
fuses the two views with structure-constrained bidirectional cross-attention, and pre-trains the backbone with an
augmentation-free contrastive objective (NT-Xent). The pre-trained backbone is fine-tuned for molecular property
prediction; the constrained attention weights link each atom to the substructures containing it.

![SJoINT overview](images/SJoINT_overview.svg)

## Layout

```
checkpoints/   pre-trained backbone (SJoINT_zinc250k_pretrained.pt)
data/          raw inputs + JT vocabulary (processed tensors are built locally; see data/README.md)
preprocess/    raw SMILES/CSV → .pt tensors
pretrain/      backbone pre-training (model.py, utils.py, train.py)
finetune/      fine-tuning: model.py, dataset.py, utils.py, moleculenet/, moleculeace/
```

## Install

```bash
conda create -n sjoint python=3.12 -y && conda activate sjoint
pip install -r requirements.txt
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

## Usage

```bash
# 1. preprocess (see preprocess/README.md)
bash preprocess/run.sh pretrain
bash preprocess/run.sh moleculenet                  # MODE=scaffold for the scaffold split
bash preprocess/run.sh moleculeace

# 2. pre-train (optional; the released backbone is checkpoints/SJoINT_zinc250k_pretrained.pt)
python pretrain/train.py --data_path data/processed/zinc250k --vocab_path data/vocab/vocab_zinc.json \
    --checkpoint_dir checkpoints/backbone --learning_rate 1e-4 --temperature 0.07 --batch_size 256 \
    --weight_decay 5e-5 --warmup_epochs 10 --max_epochs 100 --seed 42

# 3. fine-tune (see finetune/README.md)
python finetune/moleculenet/train.py --dataset BBBP
python finetune/moleculeace/train.py --task CHEMBL204_Ki --fold 0
```

A backbone must be used with the vocabulary it was pre-trained with; a new vocabulary (`VOCAB_PATH`) requires
re-tensorizing the data and pre-training a new backbone.

## Data

Pre-training: ZINC250K ([ZINC](https://zinc.docking.org), Irwin & Shoichet, *J. Chem. Inf. Model.* 2005).
Fine-tuning: [MoleculeNet](https://moleculenet.org) (Wu et al., *Chem. Sci.* 2018) and
[MoleculeACE](https://github.com/molML/MoleculeACE) (van Tilborg et al., *J. Chem. Inf. Model.* 2022).
The raw files under `data/raw/` are redistributed from these sources; please cite them when using this repository.
