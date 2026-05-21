# SJoINT — Substructure-driven JOINT molecular representation

SJoINT learns molecular representations by jointly encoding two graph views of
the same molecule:

1. **Molecular graph** — nodes are atoms, edges are bonds.
2. **Junction tree (JT)** — nodes are substructures (rings, fused ring systems,
   non-ring bonds), edges are shared atoms between them.

The two views are fused by a stack of **structure-constrained bidirectional
cross-attention blocks**: each substructure attends only to its constituent
atoms, and each atom only to substructures it belongs to. Pre-training is
**augmentation-free** — the atom view and JT view of the *same* molecule form
a natural positive pair under an NT-Xent contrastive loss. Downstream
prediction uses a **two-stage fine-tuning** protocol (frozen backbone → full
joint fine-tune).

> Status: pre-print companion repository. Pre-trained checkpoints, full
> MoleculeNet splits, and paper figures are released separately via
> [Releases](../../releases) once the manuscript is public. This repository
> contains the source code for the three pipeline stages
> (preprocess / pretrain / finetune) only.

---

## Repository layout

```
SJoINT/
├── preprocess/            SMILES → .pt tensors (pretrain / MoleculeNet / MoleculeACE)
├── pretrain/              Contrastive pre-training on ZINC250K
├── finetune/              Two-stage fine-tuning on MoleculeNet (random / scaffold) & MoleculeACE
├── environment.yml        Conda environment (verified on NVIDIA B200 / sm_100)
├── environment-lock.yml   Fully-pinned reproduction snapshot
├── requirements.txt       Pip alternative (CUDA wheel selection left to user)
├── CITATION.cff           Citation metadata
└── LICENSE
```

---

## Installation

### Conda (recommended)

```bash
mamba env create -f environment.yml
mamba activate sj
```

For exact-reproduction:

```bash
mamba env create -f environment-lock.yml
```

### Pip (no Conda)

```bash
python -m venv .venv && source .venv/bin/activate
# Install the CUDA-matching PyTorch wheel first (see https://pytorch.org/),
# then PyG against that PyTorch build, then the lightweight extras:
pip install -r requirements.txt
```

### Hardware note

The code is tested on NVIDIA B200 (sm_100) with PyTorch 2.7+ / CUDA 12.4+.
On other GPUs you may need a different PyTorch wheel. The model itself is
small (hidden dim 32, ~159 K parameters); a single mid-range GPU is enough
for fine-tuning, and pre-training on ZINC250K (~100 epochs) runs in under a
day on a single recent GPU.

---

## Pipeline

The three stages each have their own README with detailed CLI / env-var
reference:

- [`preprocess/README.md`](preprocess/README.md) — SMILES → `.pt`
- [`pretrain/README.md`](pretrain/README.md) — contrastive pre-training on ZINC250K

`preprocess/*` modules are invoked via `python -m preprocess.<sub>.<script>`
(package-style); `pretrain/*` and `finetune/*` are invoked as standalone
scripts (`python pretrain/train.py …`, `python finetune/train.py …`).

### Pre-training

```bash
python pretrain/train.py \
    --data_path data/ZINC250K \
    --checkpoint_dir pretrain/checkpoints/sjoint \
    --learning_rate 5e-4 --weight_decay 5e-5 --batch_size 512 \
    --max_epochs 100 --gpu_id 0
```

See [`pretrain/README.md`](pretrain/README.md) for the full HP sweep launcher
(`run_pretrain_sweep.py`) and all CLI arguments.

### Fine-tuning

Three benchmarks are supported. All consume the same `best_model.ckpt`
produced by the pretrain stage.

```bash
# 1) MoleculeNet — random split (defaults to data/MoleculeNet/random_split)
python finetune/run.py \
    --dataset BBBP --seed 1 \
    --pretrain_ckpt /path/to/best_model.ckpt --gpu_id 0

# 2) MoleculeNet — scaffold split (override --data_base)
python finetune/train.py \
    --dataset BBBP --seed 1 \
    --pretrain_ckpt /path/to/best_model.ckpt --gpu_id 0 \
    --data_base data/MoleculeNet/split/scaffold

# 3) MoleculeACE (van Tilborg 2022 canonical split, cliff metrics)
python finetune/run_moleculeace.py --datasets CHEMBL204_Ki CHEMBL234_Ki
```

Per-dataset best hyperparameters (10-seed × 9-dataset search) live in
[`finetune/configs.py`](finetune/configs.py) and are consumed by `run.py`
automatically. To reproduce the full protocol (9 datasets × 10 seeds ×
{random, scaffold}), loop over `--dataset` / `--seed` / `--data_base`
yourself; the per-job entry point is the single-shot `python finetune/run.py
…` command above.

### Data not shipped here

Large public corpora are referenced by download instructions rather than
committed:

| Corpus | Where to get it |
|---|---|
| ZINC250K (pre-training) | <http://files.docking.org/zinc/> or <https://github.com/wengong-jin/icml18-jtnn> |
| MoleculeNet (fine-tuning) | <https://moleculenet.org/datasets-1> |
| MoleculeACE (cliff benchmark) | <https://github.com/molML/MoleculeACE> |

Place them under `data/` per the layout in
[`preprocess/README.md`](preprocess/README.md).

---

## What is *not* in this repository

- **Pre-trained checkpoints.** Released separately via GitHub Releases /
  Zenodo once the manuscript is public.
- **Raw / processed datasets.** ZINC250K, MoleculeNet `.pt` tensors, and
  ChEMBL corpora are not committed; see download instructions above.
- **Paper figures, ablations, and analysis pipelines.** Out of scope for
  this code release.
- **Internal experiment logs and cluster orchestrators.** Hyperparameter
  sweep logs, training logs, watchdog scripts, multi-GPU launchers and
  per-sweep result aggregators are excluded; the canonical entry points
  (`pretrain/train.py`, `finetune/train.py`, `finetune/run.py`,
  `finetune/run_moleculeace.py`) are sufficient to reproduce results from
  scratch.

---

## License

Released under the MIT License (see [`LICENSE`](LICENSE)).

## Citation

If you use this code, please cite the work using the metadata in
[`CITATION.cff`](CITATION.cff). The citation block will be finalised once the
paper is accepted.
