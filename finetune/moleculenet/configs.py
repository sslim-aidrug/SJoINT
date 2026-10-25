"""Fine-tuning hyperparameter search space and defaults (MoleculeNet and MoleculeACE)."""

SEARCH_SPACE = {
    "learning_rate": [1e-4, 2e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "batch_size": [32, 64, 128],
    "num_head_layers": [1, 2, 3, 4],
    "head_hidden": [32, 64, 128],
    "head_dropout": [0.0, 0.1, 0.2, 0.3],
    "stage1_epochs": [0, 10, 20, 30, 40, 50],
    "backbone_lr_ratio": [0.03, 0.1, 0.3, 1.0],
    "head_norm": ["layer", "batch", "none"],
    "head_activation": ["relu", "gelu"],
    "backbone_dropout": [-1.0, 0.0],
    "freeze_bn": [False, True],
    "preserve_optimizer_state": [False, True],
}

DEFAULT_HP = {  # BBBP, random split
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "num_head_layers": 3,
    "head_hidden": 64,
    "head_dropout": 0.0,
    "stage1_epochs": 10,
    "backbone_lr_ratio": 0.3,
}
