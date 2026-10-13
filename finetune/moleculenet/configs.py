"""Per-dataset fine-tuning hyper-parameters, keyed by split.

`DEFAULT_HP[split][dataset]` gives the hyper-parameters for a (split, dataset).
`train.py` is CLI-driven and does not read this table automatically; pass values
on the CLI or import `DEFAULT_HP[split][dataset]`.
"""

_RANDOM_HP = {
    # ── Classification (metric: ROC-AUC ↑) ───────────────
    "BBBP":      {"stage1_epochs": 20, "learning_rate": 5e-4, "weight_decay": 1e-5,
            "batch_size": 64, "num_head_layers": 3, "head_hidden": 32, "head_dropout": 0.1},
    "BACE":      {"stage1_epochs": 10, "learning_rate": 5e-4, "weight_decay": 1e-4,
            "batch_size": 128, "num_head_layers": 2, "head_hidden": 64, "head_dropout": 0.3},
    "ClinTox":   {"stage1_epochs": 50, "learning_rate": 5e-3, "weight_decay": 1e-5,
               "batch_size": 128, "num_head_layers": 3, "head_hidden": 64, "head_dropout": 0.0},
    "SIDER":     {"stage1_epochs": 40, "learning_rate": 1e-2, "weight_decay": 1e-4,
             "batch_size": 32, "num_head_layers": 3, "head_hidden": 128, "head_dropout": 0.0},
    "Tox21":     {"stage1_epochs": 10, "learning_rate": 5e-4, "weight_decay": 1e-5,
             "batch_size": 128, "num_head_layers": 2, "head_hidden": 128, "head_dropout": 0.2},
    "ToxCast":   {"stage1_epochs": 20, "learning_rate": 5e-4, "weight_decay": 1e-5,
               "batch_size": 32, "num_head_layers": 2, "head_hidden": 64, "head_dropout": 0.2},
    # ── Regression (metric: RMSE ↓) ───────────────
    "FreeSolv":  {"stage1_epochs": 40, "learning_rate": 1e-2, "weight_decay": 1e-4,
                "batch_size": 128, "num_head_layers": 2, "head_hidden": 128, "head_dropout": 0.3},
    "ESOL":      {"stage1_epochs": 40, "learning_rate": 1e-2, "weight_decay": 1e-4,
            "batch_size": 128, "num_head_layers": 2, "head_hidden": 128, "head_dropout": 0.3},
    "Lipophilicity":{"stage1_epochs": 30, "learning_rate": 5e-3, "weight_decay": 1e-4,
                     "batch_size": 64, "num_head_layers": 2, "head_hidden": 128, "head_dropout": 0.1},
}

_SCAFFOLD_HP = {
    # ── Classification (metric: ROC-AUC ↑) ───────────────
    "BBBP":      {"stage1_epochs": 30, "learning_rate": 5e-4, "weight_decay": 1e-5,
            "batch_size": 128, "num_head_layers": 3, "head_hidden": 64, "head_dropout": 0.0},
    "BACE":      {"stage1_epochs": 30, "learning_rate": 5e-4, "weight_decay": 1e-5,
            "batch_size": 128, "num_head_layers": 3, "head_hidden": 64, "head_dropout": 0.0},
    "ClinTox":   {"stage1_epochs": 50, "learning_rate": 1e-2, "weight_decay": 1e-5,
               "batch_size": 64, "num_head_layers": 3, "head_hidden": 128, "head_dropout": 0.0},
    "SIDER":     {"stage1_epochs": 40, "learning_rate": 1e-2, "weight_decay": 1e-4,
             "batch_size": 128, "num_head_layers": 2, "head_hidden": 128, "head_dropout": 0.3},
    "Tox21":     {"stage1_epochs": 40, "learning_rate": 5e-3, "weight_decay": 1e-5,
             "batch_size": 32, "num_head_layers": 4, "head_hidden": 128, "head_dropout": 0.2},
    "ToxCast":   {"stage1_epochs": 40, "learning_rate": 1e-2, "weight_decay": 1e-4,
               "batch_size": 128, "num_head_layers": 2, "head_hidden": 128, "head_dropout": 0.3},
    # ── Regression (metric: RMSE ↓) ───────────────
    "FreeSolv":  {"stage1_epochs": 50, "learning_rate": 1e-2, "weight_decay": 1e-5,
                "batch_size": 64, "num_head_layers": 3, "head_hidden": 128, "head_dropout": 0.0},
    "ESOL":      {"stage1_epochs": 50, "learning_rate": 1e-3, "weight_decay": 1e-5,
            "batch_size": 32, "num_head_layers": 1, "head_hidden": 128, "head_dropout": 0.3},
    "Lipophilicity":{"stage1_epochs": 20, "learning_rate": 5e-3, "weight_decay": 1e-4,
                     "batch_size": 128, "num_head_layers": 4, "head_hidden": 128, "head_dropout": 0.1},
}

DEFAULT_HP = {
    "random": _RANDOM_HP,
    "scaffold": _SCAFFOLD_HP,
}


def get_hp(dataset, split="random"):
    """Return reference hyper-parameters for a (dataset, split), or None."""
    return DEFAULT_HP.get(split, {}).get(dataset)
