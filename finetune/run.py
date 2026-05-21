"""Run fine-tuning with best hyperparameters from configs.py."""
from __future__ import annotations
import argparse, os, subprocess, sys, time

from configs import BEST_HP

PYTHON = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train.py")

FIXED = dict(max_epochs=200, early_stop_patience=40, num_workers=2)


def _build_cmd(dataset, seed, hp, pretrain_ckpt, gpu_id, variant=""):
    cmd = [PYTHON, "-u", TRAIN_SCRIPT,
           "--dataset", dataset,
           "--seed", str(seed),
           "--gpu_id", str(gpu_id),
           "--pretrain_ckpt", pretrain_ckpt]
    if variant:
        cmd.extend(["--variant", variant])
    for k, v in hp.items():
        cmd.extend([f"--{k}", str(v)])
    for k, v in FIXED.items():
        cmd.extend([f"--{k}", str(v)])
    return cmd


def _result_exists(dataset, seed, hp, variant=""):
    from train import DEFAULT_RESULTS_BASE
    hp_tag = (f"s1e{hp['stage1_epochs']}_lr{hp['learning_rate']}_wd{hp['weight_decay']}"
              f"_bs{hp['batch_size']}_L{hp['num_head_layers']}"
              f"_h{hp['head_hidden']}_d{hp['head_dropout']}")
    parts = [DEFAULT_RESULTS_BASE]
    if variant:
        parts.append(variant)
    parts += [dataset, f"seed{seed}", hp_tag, "results.json"]
    return os.path.exists(os.path.join(*parts))


def main():
    parser = argparse.ArgumentParser(description="SJoINT fine-tuning with best HP")
    parser.add_argument("--dataset", type=str, default="BBBP")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds (e.g. --seeds 1 2 3)")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--pretrain_ckpt", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    datasets = list(BEST_HP.keys()) if args.all else [args.dataset]
    seeds = args.seeds or [args.seed]

    # Build job queue
    queue = []
    for ds in datasets:
        if ds not in BEST_HP:
            print(f"  [SKIP] {ds}: no config in BEST_HP")
            continue
        hp = BEST_HP[ds]
        for seed in seeds:
            if _result_exists(ds, seed, hp, args.variant):
                continue
            queue.append((ds, seed, hp))

    total = len(datasets) * len(seeds)
    print("=" * 60)
    print("  SJoINT Fine-tuning")
    print(f"  Datasets : {datasets}")
    print(f"  Seeds    : {seeds}")
    print(f"  Pretrain : {args.pretrain_ckpt}")
    print(f"  Total: {total} (pending {len(queue)}, done {total - len(queue)})")
    print("=" * 60)

    if args.dry_run:
        for ds, seed, hp in queue:
            print(f"    {ds}/seed{seed}")
        return

    for i, (ds, seed, hp) in enumerate(queue, 1):
        print(f"\n  [{i}/{len(queue)}] {ds}/seed{seed}", flush=True)
        cmd = _build_cmd(ds, seed, hp, args.pretrain_ckpt, args.gpu_id, args.variant)
        t0 = time.time()
        result = subprocess.run(cmd, cwd=SCRIPT_DIR)
        elapsed = time.time() - t0
        status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
        print(f"  {status} ({elapsed:.0f}s)", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
