#!/usr/bin/env python3
"""
Aggregate per-worker metric CSVs into a global summary.

Run this script after an experiment to compare worker performance and
compute global statistics. Useful for comparing different configurations
or studying what changes when num_workers varies.

Usage:
    python scripts/aggregate_metrics.py
    python scripts/aggregate_metrics.py --data-root data/femnist  # custom path

Input:
    data/femnist/worker_*/metrics.csv      (one file per worker, required)
    data/femnist/worker_*/model_final.pt   (final checkpoints, optional)

Output (printed to stdout):
    Per-round table: round | mean_acc | std_acc | min_acc | max_acc | workers_reporting
    Per-worker summary: final accuracy, convergence round, avg peers contacted
    Weight divergence: pairwise L2 distance between final model weights (if checkpoints found)

Output (saved to disk):
    data/femnist/global_metrics.csv  — per-round aggregated stats
    data/femnist/summary.txt         — human-readable summary
"""
import argparse
import csv
import glob
import json
import os
import statistics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_worker_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aggregate per-worker metrics.")
    parser.add_argument(
        "--data-root",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "femnist"),
        help="Root directory containing worker_* subdirectories",
    )
    args = parser.parse_args()

    # Discover worker CSV files
    pattern = os.path.join(args.data_root, "worker_*", "metrics.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"No metrics files found matching: {pattern}")
        print("Run an experiment first, then re-run this script.")
        raise SystemExit(1)

    print(f"Found {len(csv_files)} worker metrics file(s):\n")
    for p in csv_files:
        print(f"  {p}")
    print()

    # Load all worker data
    all_rows: list[dict] = []
    for path in csv_files:
        rows = load_worker_csv(path)
        all_rows.extend(rows)
        print(f"  Worker {rows[0]['worker_id'] if rows else '?'}: {len(rows)} rounds logged")
    print()

    # Group rows by round number
    rounds: dict[int, list[dict]] = {}
    for row in all_rows:
        r = int(row["round"])
        rounds.setdefault(r, []).append(row)

    # ---------------------------------------------------------------------------
    # Per-round global statistics
    # ---------------------------------------------------------------------------
    print("=" * 75)
    print(f"{'Round':>6}  {'Mean Acc':>9}  {'Std Acc':>8}  {'Min Acc':>8}  {'Max Acc':>8}  {'Workers':>7}")
    print("=" * 75)

    global_rows = []
    for round_num in sorted(rounds.keys()):
        entries = rounds[round_num]
        accs = [float(e["val_accuracy"]) for e in entries]
        losses = [float(e["val_loss"]) for e in entries]
        mean_acc = statistics.mean(accs)
        std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
        min_acc = min(accs)
        max_acc = max(accs)
        mean_loss = statistics.mean(losses)
        n_workers = len(entries)

        print(
            f"{round_num:>6}  {mean_acc:>9.4f}  {std_acc:>8.4f}  "
            f"{min_acc:>8.4f}  {max_acc:>8.4f}  {n_workers:>7}"
        )
        global_rows.append({
            "round": round_num,
            "mean_accuracy": round(mean_acc, 6),
            "std_accuracy": round(std_acc, 6),
            "min_accuracy": round(min_acc, 6),
            "max_accuracy": round(max_acc, 6),
            "mean_val_loss": round(mean_loss, 6),
            "workers_reporting": n_workers,
        })
    print("=" * 75)
    print()

    # ---------------------------------------------------------------------------
    # Per-worker summary
    # ---------------------------------------------------------------------------
    worker_rows: dict[str, list[dict]] = {}
    for row in all_rows:
        worker_rows.setdefault(row["worker_id"], []).append(row)

    print("Per-worker summary:")
    print("-" * 75)
    print(f"{'Worker':>8}  {'Rounds':>7}  {'Final Acc':>10}  {'Best Acc':>9}  {'Avg Peers':>10}  {'Avg Nbrs':>9}")
    print("-" * 75)

    summary_lines = []
    for wid, rows in sorted(worker_rows.items()):
        rows_sorted = sorted(rows, key=lambda r: int(r["round"]))
        final_acc = float(rows_sorted[-1]["val_accuracy"])
        best_acc = max(float(r["val_accuracy"]) for r in rows_sorted)
        avg_peers = statistics.mean(float(r["peers_contacted"]) for r in rows_sorted)
        avg_nbrs = statistics.mean(float(r["neighbors_aggregated"]) for r in rows_sorted)
        n_rounds = len(rows_sorted)

        line = (
            f"  Worker {wid:>2}: {n_rounds} rounds | "
            f"final_acc={final_acc:.4f} | best_acc={best_acc:.4f} | "
            f"avg_peers_contacted={avg_peers:.2f} | avg_neighbors_aggregated={avg_nbrs:.2f}"
        )
        summary_lines.append(line)
        print(f"{wid:>8}  {n_rounds:>7}  {final_acc:>10.4f}  {best_acc:>9.4f}  {avg_peers:>10.2f}  {avg_nbrs:>9.2f}")
    print("-" * 75)
    print()

    # Communication volume estimate: each sent message ≈ model_size_bytes
    # (approximate — actual size varies slightly with serialization overhead)
    total_sent = sum(int(r["peers_contacted"]) for r in all_rows)
    print(f"Total gossip messages sent across all workers and rounds: {total_sent}")
    print()

    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------
    global_csv_path = os.path.join(args.data_root, "global_metrics.csv")
    global_fields = ["round", "mean_accuracy", "std_accuracy", "min_accuracy",
                     "max_accuracy", "mean_val_loss", "workers_reporting"]
    with open(global_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=global_fields)
        w.writeheader()
        w.writerows(global_rows)
    print(f"Global metrics saved to: {global_csv_path}")

    # ---------------------------------------------------------------------------
    # Weight divergence (requires final checkpoints — optional)
    # ---------------------------------------------------------------------------
    checkpoint_paths = sorted(glob.glob(
        os.path.join(args.data_root, "worker_*", "model_final.pt")
    ))
    if len(checkpoint_paths) >= 2:
        print("\nModel weight divergence (pairwise L2 distance of final weights):")
        print("-" * 55)
        try:
            import torch
            checkpoints = {}
            for path in checkpoint_paths:
                worker_dir = os.path.basename(os.path.dirname(path))
                state = torch.load(path, map_location="cpu", weights_only=True)
                # Flatten all float parameters into a single 1-D vector
                flat = torch.cat([v.float().flatten() for v in state.values()
                                  if isinstance(v, torch.Tensor) and v.is_floating_point()])
                checkpoints[worker_dir] = flat

            workers_sorted = sorted(checkpoints.keys())
            all_distances = []
            for i, wa in enumerate(workers_sorted):
                for wb in workers_sorted[i + 1:]:
                    dist = (checkpoints[wa] - checkpoints[wb]).norm().item()
                    all_distances.append(dist)
                    print(f"  {wa} ↔ {wb}: L2 = {dist:.4f}")

            mean_dist = statistics.mean(all_distances)
            print(f"\n  Mean pairwise L2 distance: {mean_dist:.4f}")
            print(
                "  → Small distance = models converged toward the same solution (FL working)\n"
                "  → Large distance = models diverged (try more rounds or more gossip peers)"
            )
            summary_lines.append(f"\nMean pairwise L2 weight distance: {mean_dist:.4f}")
        except Exception as exc:
            print(f"  Could not compute weight divergence: {exc}")
    else:
        print("\nNo final checkpoints found — skipping weight divergence analysis.")
        print("  (checkpoints are saved automatically at the end of each experiment)")

    print()

    # ---------------------------------------------------------------------------
    # Test set results (present only when use_test_set: true in config.yaml)
    # ---------------------------------------------------------------------------
    test_result_paths = sorted(glob.glob(
        os.path.join(args.data_root, "worker_*", "test_result.json")
    ))
    if test_result_paths:
        print("Test set results (unbiased — evaluated once after training):")
        print("-" * 55)
        test_accs = []
        for path in test_result_paths:
            with open(path) as f:
                r = json.load(f)
            print(f"  Worker {r['worker_id']:>2}: test_accuracy={r['test_accuracy']:.4f}  test_loss={r['test_loss']:.4f}")
            test_accs.append(r["test_accuracy"])
            summary_lines.append(
                f"  Worker {r['worker_id']}: test_accuracy={r['test_accuracy']:.4f}  test_loss={r['test_loss']:.4f}"
            )
        mean_test = statistics.mean(test_accs)
        std_test = statistics.stdev(test_accs) if len(test_accs) > 1 else 0.0
        print(f"\n  Mean test accuracy: {mean_test:.4f}  (std={std_test:.4f})")
        summary_lines.append(f"\nMean test accuracy: {mean_test:.4f}  std={std_test:.4f}")
        print("-" * 55)
        print()

    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------
    summary_path = os.path.join(args.data_root, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Experiment summary — {len(csv_files)} workers\n")
        f.write("=" * 60 + "\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write(f"\nTotal gossip messages sent: {total_sent}\n")
    print(f"Summary saved to:       {summary_path}")


if __name__ == "__main__":
    main()
