"""Aggregate MoleculeACE final results into a per-task table + macro average.

Reads `<results>/<task>/rep<n>/results.json` and writes `moleculeace_results.csv`
(per-task RMSE / RMSE_cliff mean +/- std over replicates, plus a MEAN row).

    python finetune/moleculeace/aggregate.py [results_dir] [out_csv]
"""
import os
import sys
import csv
import glob
import json
import statistics as st

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "results")
    out_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_HERE, "moleculeace_results.csv")

    rows, Rm, Cm = [], [], []
    for task_dir in sorted(glob.glob(os.path.join(results_dir, "*"))):
        task = os.path.basename(task_dir)
        rr, cc = [], []
        for rj in sorted(glob.glob(os.path.join(task_dir, "rep*", "results.json"))):
            te = json.load(open(rj))["test"]
            rr.append(te["rmse"])
            if te.get("rmse_cliff") is not None:
                cc.append(te["rmse_cliff"])
        if not rr:
            continue
        rm, rs = sum(rr) / len(rr), (st.pstdev(rr) if len(rr) > 1 else 0.0)
        cm, cs = (sum(cc) / len(cc), st.pstdev(cc) if len(cc) > 1 else 0.0) if cc else (float("nan"), 0.0)
        rows.append([task, round(rm, 4), round(rs, 4), round(cm, 4), round(cs, 4), len(rr)])
        Rm.append(rm); Cm.append(cm)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "rmse_mean", "rmse_std", "rmse_cliff_mean", "rmse_cliff_std", "n_replicate"])
        w.writerows(rows)
        if Rm:
            w.writerow(["MEAN", round(sum(Rm) / len(Rm), 4), round(st.pstdev(Rm), 4),
                        round(sum(Cm) / len(Cm), 4), round(st.pstdev(Cm), 4), len(Rm)])
    if Rm:
        print(f"[aggregate] {len(rows)} tasks -> {out_csv}")
        print(f"  MACRO  RMSE={sum(Rm)/len(Rm):.4f}+/-{st.pstdev(Rm):.4f}  "
              f"RMSE_cliff={sum(Cm)/len(Cm):.4f}+/-{st.pstdev(Cm):.4f}")
    else:
        print("[aggregate] no results found")


if __name__ == "__main__":
    main()
