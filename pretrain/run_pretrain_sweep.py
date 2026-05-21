"""Pretrain HP sweep across multiple GPUs.

Default grid (24 combos):
  lr  : [5e-5, 1e-4, 2e-4, 5e-4]
  wd  : [1e-5, 5e-5, 1e-4]
  bs  : [256, 512]

Default GPUs: 0, 1 with 10 processes each = 20 concurrent slots.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import time
from itertools import product

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_SCRIPT_DIR)

# Defaults (overridable via env or CLI)
PYTHON = os.environ.get("SJOINT_PYTHON", "python")
TRAIN_SCRIPT = os.path.join(_SCRIPT_DIR, "train.py")
DATA_PATH = os.environ.get(
    "SJOINT_DATA",
    os.path.join(_REPO_DIR, "data", "ZINC250K"),
)
RESULTS_BASE = os.path.join(_SCRIPT_DIR, "sweep_results")

GRID = {
    "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4],
    "weight_decay":  [1e-5, 5e-5, 1e-4],
    "batch_size":    [256, 512],
}

FIXED = {
    "max_epochs": 100,
    "warmup_epochs": 10,
    "temperature": 0.07,
    "num_ca_blocks": 3,
    "proj_dim": 32,
    "seed": 42,
    "num_workers": 8,
}


def tag(lr, wd, bs):
    return f"lr{lr}_wd{wd}_bs{bs}"


def run_pool(pending, gpus, procs_per_gpu, log_dir, compile_flag=False):
    """Distribute jobs across GPUs, up to procs_per_gpu concurrent jobs per GPU."""
    running = {g: [] for g in gpus}     # gpu -> list of (proc, info)
    cursor = ok = fail = skipped = 0
    n = len(pending)

    def reap():
        nonlocal ok, fail
        for gpu in gpus:
            for entry in list(running[gpu]):
                proc, info = entry
                if proc.poll() is None:
                    continue
                elapsed = time.time() - info["start"]
                if proc.returncode == 0:
                    ok += 1
                    s = "OK"
                else:
                    fail += 1
                    s = f"FAIL({proc.returncode})"
                done = ok + fail + skipped
                print(f"  [{done}/{n}] gpu{gpu} {s}: {info['tag']} ({elapsed/60:.1f}min)",
                      flush=True)
                running[gpu].remove(entry)

    def find_open_gpu():
        # Pick the GPU with the fewest running jobs (load balance)
        best = None
        for g in gpus:
            if len(running[g]) < procs_per_gpu:
                if best is None or len(running[g]) < len(running[best]):
                    best = g
        return best

    start = time.time()
    while True:
        reap()

        # Spawn new jobs until each GPU is at capacity
        while cursor < n:
            gpu = find_open_gpu()
            if gpu is None:
                break
            item = pending[cursor]
            cursor += 1
            ckpt_dir = item["ckpt_dir"]
            os.makedirs(ckpt_dir, exist_ok=True)

            best_ckpt = os.path.join(ckpt_dir, "best_model.ckpt")
            if os.path.exists(best_ckpt):
                skipped += 1
                done = ok + fail + skipped
                print(f"  [{done}/{n}]      SKIP: {item['tag']} (already done)",
                      flush=True)
                continue

            cmd = [
                PYTHON, "-u", TRAIN_SCRIPT,
                "--data_path", DATA_PATH,
                "--gpu_id", str(gpu),
                "--checkpoint_dir", ckpt_dir,
                "--learning_rate", str(item["lr"]),
                "--weight_decay", str(item["wd"]),
                "--batch_size", str(item["bs"]),
                "--max_epochs", str(FIXED["max_epochs"]),
                "--warmup_epochs", str(FIXED["warmup_epochs"]),
                "--temperature", str(FIXED["temperature"]),
                "--num_ca_blocks", str(FIXED["num_ca_blocks"]),
                "--proj_dim", str(FIXED["proj_dim"]),
                "--seed", str(FIXED["seed"]),
                "--num_workers", str(FIXED["num_workers"]),
                "--bf16",
            ]
            if compile_flag:
                cmd.append("--compile")

            log_path = os.path.join(log_dir, f"{item['tag']}.log") if log_dir else None
            stdout = open(log_path, "w") if log_path else subprocess.DEVNULL
            proc = subprocess.Popen(
                cmd, stdout=stdout, stderr=subprocess.STDOUT, cwd=_SCRIPT_DIR)
            running[gpu].append((proc, {
                "tag": item["tag"], "start": time.time(), "gpu": gpu, "log": stdout}))

        if all(not v for v in running.values()) and cursor >= n:
            break
        time.sleep(2.0)

    print(f"\n  Done: {ok} OK, {fail} FAIL, {skipped} SKIP, "
          f"{(time.time()-start)/60:.1f}min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1",
                        help="Comma-separated GPU ids (default: 0,1)")
    parser.add_argument("--procs_per_gpu", type=int, default=10,
                        help="Concurrent training procs per GPU (default: 10)")
    parser.add_argument("--results_dir", default=RESULTS_BASE,
                        help="Where to write per-config checkpoints")
    parser.add_argument("--log_dir", default=None,
                        help="Optional dir to capture per-config stdout/stderr logs "
                             "(default: <results_dir>/_logs)")
    parser.add_argument("--compile", action="store_true",
                        help="Pass --compile to train.py (Inductor)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    if not gpus:
        raise ValueError("No GPUs specified")

    log_dir = args.log_dir or os.path.join(args.results_dir, "_logs")
    os.makedirs(log_dir, exist_ok=True)

    keys = list(GRID.keys())
    combos = list(product(*GRID.values()))

    pending = []
    for vals in combos:
        hp = dict(zip(keys, vals))
        t = tag(hp["learning_rate"], hp["weight_decay"], hp["batch_size"])
        pending.append({
            "tag": t,
            "lr": hp["learning_rate"],
            "wd": hp["weight_decay"],
            "bs": hp["batch_size"],
            "ckpt_dir": os.path.join(args.results_dir, t),
        })

    print("=" * 80)
    print(f"  Pretrain HP Sweep")
    print(f"  Combos     : {len(combos)} "
          f"(lr×wd×bs = {len(GRID['learning_rate'])}×"
          f"{len(GRID['weight_decay'])}×{len(GRID['batch_size'])})")
    print(f"  GPUs       : {gpus}  ×  {args.procs_per_gpu} procs each "
          f"= {len(gpus) * args.procs_per_gpu} concurrent")
    print(f"  Python     : {PYTHON}")
    print(f"  Data path  : {DATA_PATH}")
    print(f"  Results    : {args.results_dir}")
    print(f"  Logs       : {log_dir}")
    print("=" * 80)

    if args.dry_run:
        for p in pending:
            print(f"  {p['tag']}")
        return

    run_pool(pending, gpus, args.procs_per_gpu, log_dir, compile_flag=args.compile)

    # Summary
    import json
    print(f"\n{'='*80}")
    print(f"  Pretrain Sweep Results (best val loss)")
    print(f"{'='*80}")
    for p in pending:
        log_path = os.path.join(p["ckpt_dir"], "train.log")
        best_ckpt = os.path.join(p["ckpt_dir"], "best_model.ckpt")
        if os.path.exists(best_ckpt):
            best_val = None
            if os.path.exists(log_path):
                with open(log_path) as f:
                    for line in f:
                        if "Val Loss:" in line:
                            try:
                                val_str = line.split("Val Loss:")[1].strip().split()[0]
                                best_val = float(val_str)
                            except Exception:
                                pass
            status = f"val_loss={best_val:.6f}" if best_val else "done"
            print(f"  {p['tag']:<30s}  {status}")
        else:
            print(f"  {p['tag']:<30s}  MISSING")
    print("=" * 80)


if __name__ == "__main__":
    main()
