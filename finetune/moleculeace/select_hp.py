"""Consolidate the per-task CV search results into one HP table.

Reads `<hp_cv>/<task>_hp.json` (written by hp_search.py) and writes
`moleculeace_hp.json` = {task: {hp, hp_tag, epoch_budget, cv_rmse}}, consumed by
`train.py` for the final full-train runs.

    python finetune/moleculeace/select_hp.py
"""
import os
import sys
import glob
import json

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    hp_cv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "hp_cv")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_HERE, "moleculeace_hp.json")
    table = {}
    for f in sorted(glob.glob(os.path.join(hp_cv, "*_hp.json"))):
        d = json.load(open(f))
        sel = d["selected"]
        table[d["task"]] = {
            "hp": sel["hp"], "hp_tag": sel["hp_tag"],
            "epoch_budget": sel["epoch_budget"],
            "cv_rmse": round(sel["cv_rmse"], 4),
        }
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"[select] {len(table)} tasks -> {out_path}")


if __name__ == "__main__":
    main()
