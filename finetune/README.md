# Fine-tuning

Downstream property prediction with a pretrained SJoINT backbone. The two
benchmarks follow **different, benchmark-native protocols** and live in separate
sub-packages; the shared training engine and model are at the top level.

```
finetune/
├── model.py            DownstreamModel (backbone + MLP head) + checkpoint loader
├── dataset.py          FinetuneDataset, collate, cross-attention mask, label z-scoring
├── engine.py           shared training primitives (two-stage loop, metrics, model factory)
├── moleculenet/        MoleculeNet benchmark
│   ├── train.py            entry point (per-split random/scaffold, early-stop by val loss)
│   ├── hp_search.py        hyper-parameter search by validation loss
│   └── configs.py          search space (SEARCH_SPACE) + representative defaults (DEFAULT_HP)
└── moleculeace/        MoleculeACE activity-cliff benchmark (van Tilborg 2022 protocol)
    ├── train.py            one fold model (train on 4 folds, early-stop on 1) -> fixed test
    ├── hp_search.py        k-fold CV hyper-parameter search on the training set
    ├── select_hp.py        add the CV-selected HP to moleculeace_hp.json
    ├── core.py             fixed-split loading, k-fold, backbone template
    ├── run_cv_search.py    orchestrate the CV search over all tasks
    ├── run_final.py        orchestrate the fold-model runs over all tasks
    └── moleculeace_hp.json HP table (representative `default` entry + per-task entries)
```

## MoleculeNet

Two-stage schedule: **Stage 1** warms up the freshly-initialised head with the
backbone frozen (`stage1_epochs`); **Stage 2** unfreezes the backbone and
fine-tunes end-to-end at `learning_rate × backbone_lr_ratio`. Each dataset ships
several random (or scaffold) splits — **one per seed** — and the best epoch is
selected by **validation loss** (`max_epochs=100`, `early_stop_patience=20`). The
backbone architecture is auto-detected from the checkpoint.

```bash
python finetune/moleculenet/train.py --dataset BBBP --seed 42 \
    --pretrain_ckpt checkpoints/SJoINT_zinc250k_pretrained.pt \
    --data_base data/processed/moleculenet/random_split
```

`--pretrain_ckpt` defaults to the released backbone; pass the path of a newly
pre-trained backbone (e.g. `checkpoints/backbone/best_model.ckpt`) to fine-tune it
instead, or `none` for a randomly initialised backbone (no-pretrain baseline). The
downstream tensors must be built with the vocabulary of the backbone being used
(see `preprocess/README.md`). `--data_base` defaults to the random split; for the
scaffold split, point it at `data/processed/moleculenet/scaffold_split`.
Hyper-parameter defaults are the representative values in `moleculenet/configs.py`.
Per-dataset values are selected by validation loss over `SEARCH_SPACE`: 24 random
combinations, each trained on three seeds, keeping the lowest mean validation loss
(test sets are not used):

```bash
python finetune/moleculenet/hp_search.py --dataset BBBP --seeds 42 43 44 \
    --data_base data/processed/moleculenet/random_split
```
The selected values (`hp_search_results/<dataset>_hp.json`) are then passed to
`train.py` on the CLI.

## MoleculeACE

The activity-cliff benchmark follows the **original protocol** (van Tilborg et
al. 2022): a single **fixed train/test split** per task; the training set is split
into **5 folds stratified by median-binarized activity**; hyper-parameters are
chosen by Gaussian-process Bayesian optimization (at most 50 combinations; Optuna
`GPSampler`) scored by this cross-validation; and **one model per fold** (trained
on 4 folds, early stopping on the held-out fold) predicts the fixed test set, with
the test metrics **averaged over the five fold models**.
Metrics are **RMSE** (full test) and **RMSE_cliff** (RMSE on the activity-cliff
subset), auto-computed from `cliff_mask.pt`.

The test set is never used for model selection: hyper-parameters come from the
cross-validation, and each fold model is early-stopped on its own held-out fold.

```bash
# 1) 5-fold CV Bayesian HP search (per task, or all tasks via run_cv_search.py)
python finetune/moleculeace/hp_search.py --task CHEMBL204_Ki
GPUS=0 PER_GPU=8 python finetune/moleculeace/run_cv_search.py

# 2) add the selected HP to moleculeace_hp.json (tasks without an entry use `default`)
python finetune/moleculeace/select_hp.py

# 3) fold models (train on 4 folds, early-stop on 1) evaluated on the fixed test set
python finetune/moleculeace/train.py --task CHEMBL204_Ki --fold 0
GPUS=0 PER_GPU=8 N_FOLDS=5 python finetune/moleculeace/run_final.py
python finetune/moleculeace/aggregate.py       # per-task average over the fold models
```

## Outputs

- **MoleculeNet** — `moleculenet/results/<dataset>/seed<n>/<hp_tag>/`:
  `results.json` (best epoch, val/test loss, test metric — ROC-AUC or RMSE),
  `test_preds.npz` (original units), `downstream.log`.
- **MoleculeACE** — `moleculeace/results/<task>/fold<k>/`: `results.json`
  (RMSE, RMSE_cliff, n_cliff, best epoch, HP), `test_preds.npz` (original units),
  `downstream.log`.

`pool_mode` selects the readout: `both` (concat JT ⊕ MOL, default), `jt_only`,
or `mol_only`.
