"""MoleculeNet hyperparameter search by validation loss.
    python finetune/moleculenet/hp_search.py --dataset BBBP --seeds 42 43 44
"""
import os
import sys
import json
import random
import argparse
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from configs import SEARCH_SPACE                                      # noqa: E402
from train import Config                                              # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
TRAIN = os.path.join(_HERE, "train.py")


def sample_combos(n, seed):
    rng, combos, seen = random.Random(seed), [], set()
    n = min(n, int(np.prod([len(v) for v in SEARCH_SPACE.values()])))
    while len(combos) < n:
        hp = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        key = tuple(sorted(hp.items()))
        if key not in seen:
            seen.add(key)
            combos.append(hp)
    return combos


def main():
    ap = argparse.ArgumentParser(description="MoleculeNet HP search (validation loss)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--n_combos", type=int, default=24)
    ap.add_argument("--search_seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--data_base", default=os.path.join(_PROJECT_ROOT, "data", "processed",
                                                         "moleculenet", "random_split"))
    ap.add_argument("--pretrain_ckpt", default="")
    ap.add_argument("--results_base", default=os.path.join(_HERE, "hp_search_results"))
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--max_epochs", type=int, default=100)
    args = ap.parse_args()

    scores = []
    combos = sample_combos(args.n_combos, args.search_seed)
    for c, hp in enumerate(combos):
        val = []
        for seed in args.seeds:
            run_dir = Config(dataset=args.dataset, seed=seed, results_base=args.results_base, **hp).output_dir
            res = os.path.join(run_dir, "results.json")
            if not os.path.exists(res):
                cmd = [sys.executable, TRAIN, "--dataset", args.dataset, "--seed", str(seed),
                       "--data_base", args.data_base, "--results_base", args.results_base,
                       "--gpu_id", str(args.gpu_id), "--max_epochs", str(args.max_epochs)]
                if args.pretrain_ckpt:
                    cmd += ["--pretrain_ckpt", args.pretrain_ckpt]
                for k, v in hp.items():
                    if isinstance(v, bool):
                        cmd += [f"--{k}"] if v else []
                    else:
                        cmd += [f"--{k}", str(v)]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            val.append(json.load(open(res))["best_val_loss"])
        tag = os.path.basename(run_dir)
        scores.append({"hp": hp, "hp_tag": tag, "val_loss": float(np.mean(val)),
                       "val_loss_std": float(np.std(val))})
        print(f"[{c + 1}/{len(combos)}] {tag}  val_loss={scores[-1]['val_loss']:.4f}", flush=True)

    scores.sort(key=lambda r: r["val_loss"])
    out = {"dataset": args.dataset, "seeds": args.seeds, "selected": scores[0], "all": scores}
    path = os.path.join(args.results_base, f"{args.dataset}_hp.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[hp-search] {args.dataset} selected {scores[0]['hp_tag']} -> {path}")


if __name__ == "__main__":
    main()
