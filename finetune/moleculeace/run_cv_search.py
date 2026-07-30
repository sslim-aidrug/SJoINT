"""Run the k-fold CV hyper-parameter search over all MoleculeACE tasks.

One subprocess per task (each does n_combos x n_folds CV runs in-process). Set
`GPUS` and `PER_GPU` via the environment.

    GPUS=3 PER_GPU=8 python finetune/moleculeace/run_cv_search.py
"""
import os
import sys
import time
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SEARCH = os.path.join(_HERE, "hp_search.py")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA_BASE = os.environ.get("DATA_BASE", os.path.join(_PROJECT_ROOT, "data", "processed", "moleculeace"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(_HERE, "hp_cv"))

GPUS = [int(g) for g in os.environ.get("GPUS", "0").split(",")]
PER_GPU = int(os.environ.get("PER_GPU", "8"))
N_FOLDS = os.environ.get("N_FOLDS", "5")
N_COMBOS = os.environ.get("N_COMBOS", "24")


def tasks():
    return sorted(d for d in os.listdir(DATA_BASE)
                  if os.path.isdir(os.path.join(DATA_BASE, d)))


def launch(task, gpu):
    cmd = [PY, "-u", SEARCH, "--task", task, "--gpu_id", "0",
           "--data_base", DATA_BASE, "--out_dir", OUT_DIR,
           "--n_folds", N_FOLDS, "--n_combos", N_COMBOS, "--num_workers", "0"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    return subprocess.Popen(cmd, env=env)


def main():
    q = [t for t in tasks()
         if not os.path.exists(os.path.join(OUT_DIR, f"{t}_hp.json"))]
    print(f"[cv-search] {len(q)} tasks, {PER_GPU}/GPU on {GPUS}", flush=True)
    running, load = {}, {g: 0 for g in GPUS}
    cur = ok = fail = 0
    t0 = time.time()
    while cur < len(q) or running:
        for pid in list(running):
            proc, task, gpu = running[pid]
            if proc.poll() is None:
                continue
            load[gpu] -= 1
            ok += proc.returncode == 0
            fail += proc.returncode != 0
            print(f"  [{ok+fail}/{len(q)}] rc={proc.returncode} {task}", flush=True)
            del running[pid]
        while cur < len(q):
            free = [g for g in GPUS if load[g] < PER_GPU]
            if not free:
                break
            g = min(free, key=lambda x: load[x])
            p = launch(q[cur], g)
            running[p.pid] = (p, q[cur], g)
            load[g] += 1
            cur += 1
        time.sleep(1.0)
    print(f"[cv-search] done: {ok} OK, {fail} FAIL, {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
