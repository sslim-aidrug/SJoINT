# Fine-tuning

Downstream property prediction with a pretrained SJoINT backbone. The two
benchmarks follow **different, benchmark-native protocols** and live in separate
sub-packages; the shared training engine and model are at the top level.

```
finetune/
├── model.py            DownstreamModel (backbone + MLP head) + checkpoint loader
├── dataset.py          FinetuneDataset, collate, cross-attention mask, label z-scoring
├── engine.py           shared training primitives (two-stage loop, metrics, model factory)
├── moleculenet/        MoleculeNet ADMET benchmark
│   ├── train.py            entry point (per-split random/scaffold, early-stop by val loss)
│   └── configs.py          per-dataset reference hyper-parameters (DEFAULT_HP)
└── moleculeace/        MoleculeACE activity-cliff benchmark (van Tilborg 2022 protocol)
    ├── train.py            final training (full train, fixed test) — one run per replicate
    ├── hp_search.py        k-fold CV hyper-parameter search on the training set
    ├── select_hp.py        consolidate CV results -> moleculeace_hp.json
    ├── core.py             fixed-split loading, k-fold, backbone template
    ├── run_cv_search.py    orchestrate the CV search over all tasks
    ├── run_final.py        orchestrate the final runs over all tasks/replicates
    └── moleculeace_hp.json per-task selected HP + epoch budget
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

`--pretrain_ckpt` defaults to the released backbone and `--data_base` to the
random split. For the scaffold split, point `--data_base` at
`data/processed/moleculenet/scaffold_split`. Per-dataset hyper-parameters are in
`moleculenet/configs.py` (`DEFAULT_HP[split][dataset]`). Omitting a pretrain
checkpoint trains a randomly-initialised backbone as a no-pretrain baseline.

## MoleculeACE

The activity-cliff benchmark follows the **original protocol** (van Tilborg et
al. 2022): a single **fixed train/test split** per task; hyper-parameters chosen
by **k-fold cross-validation on the training set**; and the final model
**retrained on the full training set**, evaluated once on the fixed test set.
Metrics are **RMSE** (full test) and **RMSE_cliff** (RMSE on the activity-cliff
subset), auto-computed from `cliff_mask.pt`.

There is no data resampling across runs. A `--replicate` index only changes the
**initialisation** (a repeated training run on the identical fixed split); the
reported numbers average over replicates, never over data splits.

```bash
# 1) k-fold CV HP search (per task, or all tasks via run_cv_search.py)
python finetune/moleculeace/hp_search.py --task CHEMBL204_Ki
GPUS=0 PER_GPU=8 python finetune/moleculeace/run_cv_search.py

# 2) consolidate the selected HP + epoch budgets
python finetune/moleculeace/select_hp.py

# 3) final training (full train, fixed test), averaged over replicates
python finetune/moleculeace/train.py --task CHEMBL204_Ki --replicate 0
GPUS=0 PER_GPU=8 N_REPLICATES=5 python finetune/moleculeace/run_final.py
```

## Outputs

- **MoleculeNet** — `moleculenet/results/<dataset>/seed<n>/<hp_tag>/`:
  `results.json` (best epoch, val/test loss, test metric — ROC-AUC or RMSE),
  `test_preds.npz`, `downstream.log`.
- **MoleculeACE** — `moleculeace/results/<task>/rep<n>/`: `results.json`
  (RMSE, RMSE_cliff, n_cliff, epoch budget, HP), `test_preds.npz`, `downstream.log`.

`pool_mode` selects the readout: `both` (concat JT ⊕ MOL, default), `jt_only`,
or `mol_only`.
