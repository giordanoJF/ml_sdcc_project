#!/usr/bin/env python3
"""
Compute functional-convergence spread on each worker's best-val-loss checkpoint
(model_best.pt) instead of on whatever round early stopping happened to land
on (final_global_test, used by aggregate_metrics.py). Early stopping keeps
training `patience` rounds past the best val_loss, so final_global_test can
capture post-peak drift rather than the checkpoint that is actually saved and
reported.

Standalone and read-only with respect to everything except its own output:
writes a single new file, functional_convergence_bestval.txt, inside
--data-root. Never touches metrics.csv, summary.txt, global_metrics.csv, or
any plot. Safe to point at the current run or at any already-archived run.

Usage:
    python scripts/aggregate_metrics_2.py                              # current run
    python scripts/aggregate_metrics_2.py --data-root data/femnist     # current run, explicit
    python scripts/aggregate_metrics_2.py --data-root results/<name>   # archived run

Input:
    <data-root>/worker_*/metrics.csv   (must contain a global_test_accuracy column)

Output:
    <data-root>/functional_convergence_bestval.txt
"""
import argparse
import csv
import glob
import os
import statistics


def load_worker_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "femnist"),
        help="Root directory containing worker_* subdirectories "
             "(current run, or an archived results/<name>/ directory)",
    )
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.data_root, "worker_*", "metrics.csv")))
    if not csv_files:
        print(f"No metrics files found matching: {args.data_root}/worker_*/metrics.csv")
        raise SystemExit(1)

    all_rows: list[dict] = []
    for path in csv_files:
        all_rows.extend(load_worker_csv(path))

    has_global_test = (
        "global_test_accuracy" in (all_rows[0] if all_rows else {})
        and any(r.get("global_test_accuracy", "") != "" for r in all_rows)
    )
    if not has_global_test:
        print(f"No global_test_accuracy column found in {args.data_root} — nothing to compute.")
        raise SystemExit(1)

    worker_rows: dict[str, list[dict]] = {}
    for row in all_rows:
        worker_rows.setdefault(row["worker_id"], []).append(row)

    best_ckpt_gt: dict[str, float] = {}
    for wid, rows in sorted(worker_rows.items()):
        rows_sorted = sorted(rows, key=lambda r: int(r["round"]))
        best_row = min(rows_sorted, key=lambda r: float(r["val_loss"]))
        gt_val = best_row.get("global_test_accuracy", "")
        if gt_val != "":
            best_ckpt_gt[wid] = float(gt_val)

    if len(best_ckpt_gt) <= 1:
        print("Not enough workers with a valid checkpoint to compute spread.")
        raise SystemExit(1)

    vals = list(best_ckpt_gt.values())
    mean_v = statistics.mean(vals)
    std_v = statistics.stdev(vals)
    spread = max(vals) - min(vals)
    if spread < 0.02:
        verdict = "STRONG functional convergence (spread < 2%)"
    elif spread < 0.05:
        verdict = "MODERATE functional convergence (spread 2–5%)"
    else:
        verdict = "WEAK functional convergence (spread > 5%) — models still diverge"

    lines = [
        "Functional convergence @ model_best.pt (per-worker best-val-loss checkpoint)",
        "=" * 65,
        "",
        "  (global_test_accuracy evaluated at each worker's own best-val-loss round,",
        "   i.e. the round actually saved as model_best.pt — not the final round)",
        "",
    ]
    for wid, gt_val in sorted(best_ckpt_gt.items()):
        lines.append(f"  Worker {wid:>2}: global_test@model_best={gt_val:.4f}")
    lines += [
        "",
        f"  Mean global test @ model_best.pt : {mean_v:.4f}",
        f"  Std across workers                : {std_v:.4f}",
        f"  Max spread (max - min)             : {spread:.4f}",
        "",
        f"  Verdict: {verdict}",
    ]

    out_path = os.path.join(args.data_root, "functional_convergence_bestval.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
