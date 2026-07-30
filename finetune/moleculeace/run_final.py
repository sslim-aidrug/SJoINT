"""Run the final MoleculeACE training over all tasks and replicates.

Full-train on the fixed training set with the CV-selected HP (moleculeace_hp.json)
and evaluate on the fixed test set. `--replicate` indices are training replicates
(initialisation only; the split is fixed), averaged for the reported mean +/- std.

    GPUS=3 PER_GPU=8 N_REPLICATES=5 python finetune/moleculeace/run_final.py
"""
import os
import sys
import json
import time
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TRAIN = os.path.join(_HERE, "train.py")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA_BASE = os.environ.get("DATA_BASE", os.path.join(_PROJECT_ROOT, "data", "processed", "moleculeace"))
HP_JSON = os.environ.get("HP_JSON", os.path.join(_HERE, "moleculeace_hp.json"))
RESULTS_BASE = os.environ.get("RESULTS_BASE", os.path.join(_HERE, "results"))

GPUS = [int(g) for g in os.environ.get("GPUS", "0").split(",")]
PER_GPU = int(os.environ.get("PER_GPU", "8"))
N_REPLICATES = int(os.environ.get("N_REPLICATES", "5"))


def launch(task, rep, gpu):
    cmd = [PY, "-u", TRAIN, "--task", task, "--replicate", str(rep), "--gpu_id", "0",
           "--data_base", DATA_BASE, "--hp_json", HP_JSON,
           "--results_base", RESULTS_BASE, "--num_workers", "0", "--save_best_ckpt"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    return subprocess.Popen(cmd, env=env)


def main():
    tasks = sorted(json.load(open(HP_JSON)).keys())
    q = [(t, r) for t in tasks for r in range(N_REPLICATES)
         if not os.path.exists(os.path.join(RESULTS_BASE, t, f"rep{r}", "results.json"))]
    print(f"[final] {len(q)} runs ({len(tasks)} tasks x {N_REPLICATES} reps), "
          f"{PER_GPU}/GPU on {GPUS}", flush=True)
    running, load = {}, {g: 0 for g in GPUS}
    cur = ok = fail = 0
    t0 = time.time()
    while cur < len(q) or running:
        for pid in list(running):
            proc, lbl, gpu = running[pid]
            if proc.poll() is None:
                continue
            load[gpu] -= 1
            ok += proc.returncode == 0
            fail += proc.returncode != 0
            if (ok + fail) % 20 == 0 or proc.returncode != 0:
                print(f"  [{ok+fail}/{len(q)}] rc={proc.returncode} {lbl}", flush=True)
            del running[pid]
        while cur < len(q):
            free = [g for g in GPUS if load[g] < PER_GPU]
            if not free:
                break
            g = min(free, key=lambda x: load[x])
            t, r = q[cur]
            p = launch(t, r, g)
            running[p.pid] = (p, f"{t}/rep{r}", g)
            load[g] += 1
            cur += 1
        time.sleep(1.0)
    print(f"[final] done: {ok} OK, {fail} FAIL, {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
