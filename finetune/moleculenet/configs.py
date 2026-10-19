"""Fine-tuning hyper-parameters: search space and representative defaults.

`SEARCH_SPACE` is shared by the MoleculeNet (`moleculenet/hp_search.py`) and
MoleculeACE (`moleculeace/hp_search.py`) searches. `DEFAULT_HP` holds the
representative values `train.py` uses as CLI defaults; per-dataset values are
selected by validation loss with `hp_search.py`.
"""

SEARCH_SPACE = {
    "learning_rate": [5e-4, 1e-3, 5e-3, 1e-2],
    "weight_decay": [1e-5, 1e-4],
    "batch_size": [32, 64, 128],
    "num_head_layers": [1, 2, 3, 4],
    "head_hidden": [32, 64, 128],
    "head_dropout": [0.0, 0.1, 0.2, 0.3],
    "stage1_epochs": [10, 20, 30, 40, 50],
}

DEFAULT_HP = {
    "stage1_epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "num_head_layers": 2,
    "head_hidden": 128,
    "head_dropout": 0.1,
}
