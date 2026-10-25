# Fine-tuning

```
finetune/
├── model.py            DownstreamModel (backbone + MLP head), checkpoint loader
├── dataset.py          dataset, collate, cross-attention mask, label normalization
├── utils.py            two-stage training loop, metrics, model factory
├── moleculenet/        train.py, hp_search.py, configs.py (search space, defaults)
└── moleculeace/        train.py, hp_search.py, common.py, aggregate.py, moleculeace_hp.json
```

Both benchmarks fine-tune in two stages: the prediction head is trained with the backbone frozen (`stage1_epochs`),
then the whole model is trained with the backbone at `learning_rate × backbone_lr_ratio`. The best epoch is chosen by
validation loss (`max_epochs 100`). `--pretrain_ckpt` defaults to the released backbone (`none` = random initialization).

## MoleculeNet

One random or scaffold split per seed (`data/processed/moleculenet/{random_split,scaffold_split}`); early stopping
with patience 20. Hyperparameters default to `configs.py` (`DEFAULT_HP`) and are set per dataset on the CLI:

```bash
python finetune/moleculenet/train.py --dataset BBBP --seed 42 \
    --data_base data/processed/moleculenet/random_split
```

Hyperparameter search over `configs.py` `SEARCH_SPACE` (24 random combinations × 3 seeds, lowest mean validation loss):

```bash
python finetune/moleculenet/hp_search.py --dataset BBBP --seeds 42 43 44 \
    --data_base data/processed/moleculenet/random_split
```

## MoleculeACE

Original protocol (van Tilborg et al. 2022): fixed train/test split per task; the training set is split into
5 folds stratified by median-binarized activity; hyperparameters are selected by Gaussian-process Bayesian
optimization (≤ 50 combinations) on the cross-validation RMSE; one model per fold (early stopping on the held-out
fold, patience 10) is evaluated on the fixed test set and the five results are averaged (RMSE, RMSE_cliff).

```bash
for task in $(ls data/processed/moleculeace); do
    python finetune/moleculeace/hp_search.py --task $task                      # → moleculeace_hp.json
    for fold in 0 1 2 3 4; do python finetune/moleculeace/train.py --task $task --fold $fold; done
done
python finetune/moleculeace/aggregate.py
```

## Outputs

`moleculenet/results/<dataset>/seed<n>/<hp_tag>/` and `moleculeace/results/<task>/fold<k>/`:
`results.json` (best epoch, validation loss, test metric), `test_preds.npz`, `downstream.log`.
