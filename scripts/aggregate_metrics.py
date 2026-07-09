#!/usr/bin/env python3
"""
Aggregate per-worker metric CSVs into a global summary.

Run this script after an experiment to compare worker performance and
compute global statistics. Useful for comparing different configurations
or studying what changes when num_workers varies.

Usage:
    python scripts/aggregate_metrics.py
    python scripts/aggregate_metrics.py --plot                      # also generate PNG plots
    python scripts/aggregate_metrics.py --data-root data/femnist    # custom path, live run
    python scripts/aggregate_metrics.py --data-root results/<name>  # recompute on an archived run

Input:
    <data-root>/worker_*/metrics.csv      (one file per worker, required)
    <data-root>/worker_*/model_best.pt    (best checkpoint by val_loss; archived by
                                            save_experiment.py alongside the metrics, so weight
                                            divergence is also available for archived
                                            results/<name> runs saved after that fix — older
                                            archived runs predating it will not have it)
    <data-root>/config.yaml               (used for total_rounds; falls back to the project's
                                            live config.yaml if not found under --data-root)

Output (printed to stdout):
    Per-round table  : round | mean_acc | std_acc | min_acc | max_acc [| phase timings]
    Per-worker table : final/best accuracy, total training time, avg peers, phase breakdown
    Convergence      : per-worker status (converged vs hit round limit), wall-clock time,
                       system-level convergence (time from first worker start to last end)
    Weight divergence: pairwise L2 distance between final model weights (if checkpoints found)
    Functional convergence (global test set, if enabled):
                       [A] at each worker's final round — reference only, noisy (see below)
                       [B] at each worker's model_best.pt round — primary metric
    Local test results: per-worker local_test_accuracy (only when local_test_set: true)

Output (saved to disk):
    Live run (--data-root data/femnist, the default):
        global_metrics.csv          — per-round aggregated stats (+ phase timings if present)
        summary.txt                 — human-readable summary including convergence verdict
        accuracy_over_rounds.png    — (--plot) mean accuracy ± std and per-worker curves
        loss_over_rounds.png        — (--plot) mean val loss and per-worker curves
        phase_timing.png            — (--plot) phase A/B/C durations over rounds

    Archived run (--data-root results/<name>): never overwrites anything already in that
    folder. Writes a single new file instead:
        summary_recomputed.txt      — same structured report as summary.txt above
"""
import argparse
import csv
import glob
import json
import os
import statistics

import yaml


def _plot_results(global_rows, worker_rows, data_root, has_timing, has_global_test=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots (pip install matplotlib)")
        return

    rounds = [r["round"] for r in global_rows]
    mean_accs = [r["mean_accuracy"] for r in global_rows]
    std_accs = [r["std_accuracy"] for r in global_rows]
    mean_losses = [r["mean_val_loss"] for r in global_rows]

    # --- accuracy over rounds ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, mean_accs, label="mean accuracy", linewidth=2, color="steelblue")
    ax.fill_between(
        rounds,
        [m - s for m, s in zip(mean_accs, std_accs)],
        [m + s for m, s in zip(mean_accs, std_accs)],
        alpha=0.2, color="steelblue", label="±1 std",
    )
    for wid, rows in sorted(worker_rows.items()):
        rows_s = sorted(rows, key=lambda r: int(r["round"]))
        ax.plot(
            [int(r["round"]) for r in rows_s],
            [float(r["val_accuracy"]) for r in rows_s],
            alpha=0.55, linestyle="--", label=f"worker {wid}",
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy over Rounds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(data_root, "accuracy_over_rounds.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")

    # --- loss over rounds ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, mean_losses, linewidth=2, color="tomato", label="mean val loss")
    for wid, rows in sorted(worker_rows.items()):
        rows_s = sorted(rows, key=lambda r: int(r["round"]))
        ax.plot(
            [int(r["round"]) for r in rows_s],
            [float(r["val_loss"]) for r in rows_s],
            alpha=0.55, linestyle="--", label=f"worker {wid}",
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss over Rounds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(data_root, "loss_over_rounds.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")

    # --- phase timing (only when timing columns are present) ---
    if has_timing:
        phase_b = [r["mean_phase_b_s"] for r in global_rows]
        phase_a_ms = [r["mean_phase_a_s"] * 1000 for r in global_rows]
        phase_c_ms = [r["mean_phase_c_s"] * 1000 for r in global_rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(rounds, phase_b, color="steelblue")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Seconds")
        ax1.set_title("Phase B — Local Training Duration")
        ax1.grid(True, alpha=0.3)

        ax2.plot(rounds, phase_a_ms, color="tomato", label="Phase A (aggregation)")
        ax2.plot(rounds, phase_c_ms, color="seagreen", label="Phase C (gossip push)")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Milliseconds")
        ax2.set_title("Phase A & C Duration")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        path = os.path.join(data_root, "phase_timing.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {path}")

    # --- global test accuracy over rounds (functional convergence) ---
    if has_global_test:
        fig, ax = plt.subplots(figsize=(10, 6))

        gt_rounds = [r["round"] for r in global_rows if r.get("mean_global_test_accuracy") is not None]
        gt_means = [r["mean_global_test_accuracy"] for r in global_rows if r.get("mean_global_test_accuracy") is not None]
        gt_stds = [r.get("std_global_test_accuracy", 0.0) for r in global_rows if r.get("mean_global_test_accuracy") is not None]

        if gt_rounds:
            ax.plot(gt_rounds, gt_means, label="mean global test accuracy", linewidth=2, color="darkorange")
            ax.fill_between(
                gt_rounds,
                [m - s for m, s in zip(gt_means, gt_stds)],
                [m + s for m, s in zip(gt_means, gt_stds)],
                alpha=0.2, color="darkorange", label="±1 std",
            )

        for wid, rows in sorted(worker_rows.items()):
            rows_s = sorted(rows, key=lambda r: int(r["round"]))
            wgt_rounds = [int(r["round"]) for r in rows_s if r.get("global_test_accuracy", "") != ""]
            wgt_accs = [float(r["global_test_accuracy"]) for r in rows_s if r.get("global_test_accuracy", "") != ""]
            if wgt_rounds:
                ax.plot(wgt_rounds, wgt_accs, alpha=0.55, linestyle="--", label=f"worker {wid}")

        ax.set_xlabel("Round")
        ax.set_ylabel("Global Test Accuracy")
        ax.set_title("Functional Convergence — Global Test Accuracy over Rounds\n"
                     "(writers never seen by any worker — same set for all workers)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = os.path.join(data_root, "global_test_accuracy.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_worker_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate PNG plots (accuracy, loss, phase timing) saved alongside global_metrics.csv",
    )
    args = parser.parse_args()

    # Any --data-root other than the live working directory (i.e. an archived
    # results/<name>/ folder) is treated as read-only: we never overwrite files
    # that are already there, we only ever add a new summary_recomputed.txt.
    default_data_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "femnist"
    )
    is_archived_run = os.path.abspath(args.data_root) != os.path.abspath(default_data_root)

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

    # Detect optional columns (absent in CSV files from older runs or disabled flags)
    has_timing = "phase_a_s" in (all_rows[0] if all_rows else {})
    has_global_test = (
        "global_test_accuracy" in (all_rows[0] if all_rows else {})
        and any(r.get("global_test_accuracy", "") != "" for r in all_rows)
    )

    # ---------------------------------------------------------------------------
    # Per-round global statistics
    # ---------------------------------------------------------------------------
    if has_timing:
        print("=" * 100)
        print(f"{'Round':>6}  {'Mean Acc':>9}  {'Std Acc':>8}  {'Min Acc':>8}  {'Max Acc':>8}  "
              f"{'PhaseA(s)':>10}  {'PhaseB(s)':>10}  {'PhaseC(s)':>10}  {'Workers':>7}")
        print("=" * 100)
    else:
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

        row_data = {
            "round": round_num,
            "mean_accuracy": round(mean_acc, 6),
            "std_accuracy": round(std_acc, 6),
            "min_accuracy": round(min_acc, 6),
            "max_accuracy": round(max_acc, 6),
            "mean_val_loss": round(mean_loss, 6),
            "workers_reporting": n_workers,
        }

        if has_global_test:
            gt_accs = [float(e["global_test_accuracy"]) for e in entries
                       if e.get("global_test_accuracy", "") != ""]
            if gt_accs:
                mean_gt = statistics.mean(gt_accs)
                std_gt = statistics.stdev(gt_accs) if len(gt_accs) > 1 else 0.0
                row_data["mean_global_test_accuracy"] = round(mean_gt, 6)
                row_data["std_global_test_accuracy"] = round(std_gt, 6)

        if has_timing:
            mean_pa = statistics.mean(float(e.get("phase_a_s", 0)) for e in entries)
            mean_pb = statistics.mean(float(e.get("phase_b_s", 0)) for e in entries)
            mean_pc = statistics.mean(float(e.get("phase_c_s", 0)) for e in entries)
            row_data.update({
                "mean_phase_a_s": round(mean_pa, 4),
                "mean_phase_b_s": round(mean_pb, 4),
                "mean_phase_c_s": round(mean_pc, 4),
            })
            print(
                f"{round_num:>6}  {mean_acc:>9.4f}  {std_acc:>8.4f}  "
                f"{min_acc:>8.4f}  {max_acc:>8.4f}  "
                f"{mean_pa:>10.4f}  {mean_pb:>10.4f}  {mean_pc:>10.4f}  {n_workers:>7}"
            )
        else:
            print(
                f"{round_num:>6}  {mean_acc:>9.4f}  {std_acc:>8.4f}  "
                f"{min_acc:>8.4f}  {max_acc:>8.4f}  {n_workers:>7}"
            )

        global_rows.append(row_data)

    sep = "=" * (100 if has_timing else 75)
    print(sep)
    print()

    # ---------------------------------------------------------------------------
    # Per-worker summary
    # ---------------------------------------------------------------------------
    worker_rows: dict[str, list[dict]] = {}
    for row in all_rows:
        worker_rows.setdefault(row["worker_id"], []).append(row)

    print("Per-worker summary:")
    print("-" * 75)
    print(f"{'Worker':>8}  {'Rounds':>7}  {'Final Acc':>10}  {'Best Acc':>9}  {'Total(s)':>9}  {'Avg Peers':>10}")
    print("-" * 75)

    summary_lines = []
    all_best_accs = []
    for wid, rows in sorted(worker_rows.items()):
        rows_sorted = sorted(rows, key=lambda r: int(r["round"]))
        final_acc = float(rows_sorted[-1]["val_accuracy"])
        best_acc = max(float(r["val_accuracy"]) for r in rows_sorted)
        all_best_accs.append(best_acc)
        avg_peers = statistics.mean(float(r["peers_contacted"]) for r in rows_sorted)
        avg_nbrs = statistics.mean(float(r["neighbors_aggregated"]) for r in rows_sorted)
        n_rounds = len(rows_sorted)
        total_s = sum(float(r["round_duration_s"]) for r in rows_sorted)

        line = (
            f"  Worker {wid:>2}: {n_rounds} rounds | "
            f"final_acc={final_acc:.4f} | best_acc={best_acc:.4f} | "
            f"total_training_s={total_s:.1f} | "
            f"avg_peers_contacted={avg_peers:.2f} | avg_neighbors_aggregated={avg_nbrs:.2f}"
        )
        summary_lines.append(line)
        print(f"{wid:>8}  {n_rounds:>7}  {final_acc:>10.4f}  {best_acc:>9.4f}  "
              f"{total_s:>9.1f}  {avg_peers:>10.2f}")

        if has_timing:
            avg_pa = statistics.mean(float(r.get("phase_a_s", 0)) for r in rows_sorted)
            avg_pb = statistics.mean(float(r.get("phase_b_s", 0)) for r in rows_sorted)
            avg_pc = statistics.mean(float(r.get("phase_c_s", 0)) for r in rows_sorted)
            grpc_vals = [float(r.get("grpc_mean_latency_s", 0)) for r in rows_sorted
                         if float(r.get("grpc_mean_latency_s", 0)) > 0]
            avg_grpc = statistics.mean(grpc_vals) if grpc_vals else 0.0
            timing_line = (
                f"           phase_a={avg_pa*1000:.1f}ms  "
                f"phase_b={avg_pb:.2f}s  "
                f"phase_c={avg_pc*1000:.1f}ms  "
                f"grpc_latency={avg_grpc*1000:.2f}ms/call"
            )
            print(timing_line)
            summary_lines.append(timing_line)

    print("-" * 75)

    if not all_best_accs:
        print("\n  *** No training data found — all workers logged 0 rounds ***")
        return
    mean_best = statistics.mean(all_best_accs)
    std_best = statistics.stdev(all_best_accs) if len(all_best_accs) > 1 else 0.0
    print(f"\n  *** mean_best_val_accuracy = {mean_best:.4f}  (std={std_best:.4f}) ***")
    print("  Use this metric to compare configurations across runs.")
    summary_lines.append(f"\nmean_best_val_accuracy = {mean_best:.4f}  std={std_best:.4f}")
    print()

    # Communication volume estimate: each sent message ≈ model_size_bytes
    # (approximate — actual size varies slightly with serialization overhead)
    total_sent = sum(int(r["peers_contacted"]) for r in all_rows)
    print(f"Total gossip messages sent across all workers and rounds: {total_sent}")
    print()

    # ---------------------------------------------------------------------------
    # System convergence analysis
    # ---------------------------------------------------------------------------
    # Load total_rounds from config.yaml to distinguish "converged" from "hit limit".
    # Prefer the config archived alongside an old run (--data-root results/<name>/config.yaml)
    # so this reflects the config that actually produced the data, not today's live config.
    # Falls back to the live project config.yaml (the normal case for the current run).
    # Gracefully skipped if neither is found.
    total_rounds_cfg = None
    archived_config_path = os.path.join(args.data_root, "config.yaml")
    live_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
    )
    for config_path in (archived_config_path, live_config_path):
        try:
            with open(config_path) as f:
                total_rounds_cfg = yaml.safe_load(f)["federated_learning"]["total_rounds"]
            break
        except Exception:
            continue

    print("System convergence analysis:")
    print("-" * 65)

    worker_start_times: list[float] = []
    worker_end_times:   list[float] = []
    worker_converged:   list[bool]  = []

    for wid, rows in sorted(worker_rows.items()):
        rows_sorted = sorted(rows, key=lambda r: int(r["round"]))
        last_round  = int(rows_sorted[-1]["round"])
        n_rounds    = len(rows_sorted)

        # Wall-clock times derived from per-round timestamps.
        # timestamp is written at the END of each round, so:
        #   worker_start ≈ timestamp[round=1] - round_duration_s[round=1]
        #   worker_end   = timestamp[last_round]
        t_first_end = float(rows_sorted[0]["timestamp"])
        t_first_dur = float(rows_sorted[0]["round_duration_s"])
        t_last_end  = float(rows_sorted[-1]["timestamp"])
        worker_start = t_first_end - t_first_dur
        worker_wall  = t_last_end - worker_start

        worker_start_times.append(worker_start)
        worker_end_times.append(t_last_end)

        if total_rounds_cfg is not None:
            converged = last_round < total_rounds_cfg
            status = (f"converged at round {last_round:>4}" if converged
                      else f"hit round limit ({total_rounds_cfg})")
        else:
            converged = True   # can't tell; assume converged
            status = f"stopped at round {last_round:>4}"

        worker_converged.append(converged)
        line = f"  Worker {wid:>2}: {status:<35}  wall-clock = {worker_wall:7.1f}s"
        print(line)
        summary_lines.append(line)

    # System-level: from the earliest worker start to the latest worker end.
    # Workers run in parallel, so system convergence time ≠ sum of individual times.
    system_start = min(worker_start_times)
    system_end   = max(worker_end_times)
    system_wall  = system_end - system_start
    n_conv       = sum(worker_converged)
    n_total      = len(worker_converged)
    all_converged = n_conv == n_total

    verdict = ("YES — all workers converged" if all_converged
               else f"PARTIAL — {n_conv}/{n_total} workers converged before round limit")

    print()
    print(f"  System converged  :  {verdict}")
    print(f"  System wall-clock :  {system_wall:.1f}s  "
          f"(first worker start → last worker end)")
    if n_total > 1:
        worker_walls = [worker_end_times[i] - worker_start_times[i] for i in range(n_total)]
        print(f"  Per-worker range  :  {min(worker_walls):.1f}s – {max(worker_walls):.1f}s  "
              f"(fastest – slowest)")

    conv_summary = (
        f"\nSystem convergence: {verdict}\n"
        f"System wall-clock total: {system_wall:.1f}s"
    )
    summary_lines.append(conv_summary)
    print("-" * 65)
    print()

    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------
    if is_archived_run:
        print("Archived run — skipping global_metrics.csv (would overwrite the saved copy).")
    else:
        global_csv_path = os.path.join(args.data_root, "global_metrics.csv")
        global_fields = ["round", "mean_accuracy", "std_accuracy", "min_accuracy",
                         "max_accuracy", "mean_val_loss", "workers_reporting"]
        if has_global_test:
            global_fields += ["mean_global_test_accuracy", "std_global_test_accuracy"]
        if has_timing:
            global_fields += ["mean_phase_a_s", "mean_phase_b_s", "mean_phase_c_s"]
        with open(global_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=global_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(global_rows)
        print(f"Global metrics saved to: {global_csv_path}")

    # ---------------------------------------------------------------------------
    # Weight divergence (requires final checkpoints — optional)
    # ---------------------------------------------------------------------------
    checkpoint_paths = sorted(
        glob.glob(os.path.join(args.data_root, "worker_*", "model_best.pt"))
    )
    if len(checkpoint_paths) >= 2:
        print(f"\nModel weight divergence (pairwise L2 distance — model_best.pt):")
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
    elif is_archived_run:
        print("\nWeight divergence: N/A — no model_best.pt found in this archived run")
        print("  (predates the save_experiment.py fix that archives checkpoints).")
    else:
        print("\nNo model_best.pt checkpoints found — skipping weight divergence analysis.")
        print("  (model_best.pt is saved automatically whenever val_loss improves during training)")

    print()

    # ---------------------------------------------------------------------------
    # Global test set convergence analysis (present only when global_test_set: true)
    # ---------------------------------------------------------------------------
    # The global test set contains writers never assigned to any worker. Evaluating
    # all workers on the same data at each round reveals functional convergence:
    # if workers reach the same accuracy on unseen writers, the gossip protocol has
    # driven them to the same functional solution — not just nearby parameters.
    # ---------------------------------------------------------------------------
    def _verdict(spread: float) -> str:
        if spread < 0.02:
            return "STRONG functional convergence (spread < 2%)"
        elif spread < 0.05:
            return "MODERATE functional convergence (spread 2–5%)"
        return "WEAK functional convergence (spread > 5%) — models still diverge"

    if has_global_test:
        print("Global test set — functional convergence analysis:")
        print("-" * 65)
        print("  (writers never seen by any worker — shared evaluation set)")
        print()
        summary_lines.append(
            "\nGlobal test set — functional convergence analysis "
            "(writers never seen by any worker):"
        )

        # ---------------------------------------------------------------
        # [A] At each worker's final round. Reference only: each worker stops
        # independently (async early stopping), so this compares workers at
        # arbitrary, unrelated rounds — noisy, can look better or worse than
        # the actual saved checkpoint. Kept for continuity with older runs.
        # See [B] below for the metric to use when comparing configurations.
        # ---------------------------------------------------------------
        print("  [A] At final round (each worker's own stop point) — reference only, noisy:")
        summary_lines.append(
            "\n  [A] At final round (each worker's own stop point) — reference only, noisy:"
        )
        gt_final: dict[str, float] = {}
        for wid, rows in sorted(worker_rows.items()):
            rows_sorted = sorted(rows, key=lambda r: int(r["round"]))
            gt_vals = [float(r["global_test_accuracy"]) for r in rows_sorted
                       if r.get("global_test_accuracy", "") != ""]
            if gt_vals:
                final_gt = gt_vals[-1]
                best_gt = max(gt_vals)
                gt_final[wid] = final_gt
                line = (f"      Worker {wid:>2}: final_global_test={final_gt:.4f}  "
                        f"best_global_test={best_gt:.4f}")
                print(line)
                summary_lines.append(line)

        if len(gt_final) > 1:
            vals = list(gt_final.values())
            mean_gt = statistics.mean(vals)
            std_gt = statistics.stdev(vals)
            spread_a = max(vals) - min(vals)
            verdict_a = _verdict(spread_a)
            print(f"\n      mean={mean_gt:.4f}  std={std_gt:.4f}  spread={spread_a:.4f}")
            print(f"      Verdict: {verdict_a}")
            summary_lines.append(
                f"\n      mean={mean_gt:.4f}  std={std_gt:.4f}  spread={spread_a:.4f}\n"
                f"      Verdict: {verdict_a}"
            )

        # ---------------------------------------------------------------
        # [B] At each worker's model_best.pt round (min val_loss). This is the
        # PRIMARY metric: it evaluates every worker at the checkpoint that is
        # actually saved/reported, chosen by a criterion independent of the
        # test set, so there is no post-peak drift or selection bias.
        # ---------------------------------------------------------------
        print()
        print("  [B] At model_best.pt round (min val_loss) — PRIMARY metric:")
        summary_lines.append(
            "\n  [B] At model_best.pt round (min val_loss) — PRIMARY metric:"
        )
        best_ckpt_gt: dict[str, float] = {}
        best_ckpt_extra: dict[str, dict[str, float]] = {
            "macro_precision": {}, "macro_recall": {}, "macro_f1": {}
        }
        extra_csv_keys = {
            "macro_precision": "global_test_macro_precision",
            "macro_recall": "global_test_macro_recall",
            "macro_f1": "global_test_macro_f1",
        }
        for wid, rows in sorted(worker_rows.items()):
            rows_sorted = sorted(rows, key=lambda r: int(r["round"]))
            best_row = min(rows_sorted, key=lambda r: float(r["val_loss"]))
            gt_val = best_row.get("global_test_accuracy", "")
            if gt_val != "":
                gt_val = float(gt_val)
                best_ckpt_gt[wid] = gt_val
                extra_str = ""
                for label, csv_key in extra_csv_keys.items():
                    raw = best_row.get(csv_key, "")
                    if raw != "":
                        val = float(raw)
                        best_ckpt_extra[label][wid] = val
                        extra_str += f"  {csv_key}={val:.4f}"
                line = (f"      Worker {wid:>2}: round={int(best_row['round']):>4}  "
                        f"val_loss={float(best_row['val_loss']):.4f}  "
                        f"global_test={gt_val:.4f}{extra_str}")
                print(line)
                summary_lines.append(line)
        if len(best_ckpt_gt) > 1:
            vals = list(best_ckpt_gt.values())
            mean_bgt = statistics.mean(vals)
            std_bgt = statistics.stdev(vals)
            spread_b = max(vals) - min(vals)
            verdict_b = _verdict(spread_b)
            print(f"\n      accuracy: mean={mean_bgt:.4f}  std={std_bgt:.4f}  spread={spread_b:.4f}")
            print(f"      Verdict: {verdict_b}")
            summary_lines.append(
                f"\n      accuracy: mean={mean_bgt:.4f}  std={std_bgt:.4f}  spread={spread_b:.4f}\n"
                f"      Verdict: {verdict_b}"
            )
        elif best_ckpt_gt:
            line = f"\n      accuracy: mean={next(iter(best_ckpt_gt.values())):.4f}  (single worker — no spread)"
            print(line)
            summary_lines.append(line)
        for label in ("macro_precision", "macro_recall", "macro_f1"):
            per_worker = best_ckpt_extra[label]
            if len(per_worker) > 1:
                vals_e = list(per_worker.values())
                mean_e = statistics.mean(vals_e)
                std_e = statistics.stdev(vals_e)
                spread_e = max(vals_e) - min(vals_e)
                print(f"      {label}: mean={mean_e:.4f}  std={std_e:.4f}  spread={spread_e:.4f}")
                summary_lines.append(
                    f"      {label}: mean={mean_e:.4f}  std={std_e:.4f}  spread={spread_e:.4f}"
                )
            elif per_worker:
                line = f"      {label}: mean={next(iter(per_worker.values())):.4f}  (single worker — no spread)"
                print(line)
                summary_lines.append(line)
        if best_ckpt_extra["macro_f1"] and best_ckpt_gt:
            mean_f1_check = statistics.mean(best_ckpt_extra["macro_f1"].values())
            mean_acc_check = statistics.mean(best_ckpt_gt.values())
            if mean_acc_check - mean_f1_check > 0.05:
                print(
                    "      → macro_f1 much lower than accuracy = model does well on frequent\n"
                    "        classes but poorly on rare ones (accuracy alone would hide this)"
                )
        print("-" * 65)
        print()

    # ---------------------------------------------------------------------------
    # Local test set results (present only when local_test_set: true in config.yaml)
    # ---------------------------------------------------------------------------
    # Local test writers belong to the same worker partition but were held out from
    # gradient updates. Measures generalisation on the worker's own writer population.
    # ---------------------------------------------------------------------------
    local_test_result_paths = sorted(glob.glob(
        os.path.join(args.data_root, "worker_*", "local_test_result.json")
    ))
    if local_test_result_paths:
        print("Local test set results (per-worker, evaluated once after training):")
        print("-" * 65)
        print("  (writers from the same worker partition, held out from gradient updates)")
        local_test_accs = []
        local_test_extra: dict[str, list[float]] = {
            "macro_precision": [], "macro_recall": [], "macro_f1": []
        }
        for path in local_test_result_paths:
            with open(path) as f:
                r = json.load(f)
            extra_str = ""
            for label in ("macro_precision", "macro_recall", "macro_f1"):
                val = r.get(f"local_test_{label}")  # absent in runs saved before this metric existed
                if val is not None:
                    local_test_extra[label].append(val)
                    extra_str += f"  local_test_{label}={val:.4f}"
            print(f"  Worker {r['worker_id']:>2}: local_test_accuracy={r['local_test_accuracy']:.4f}  "
                  f"local_test_loss={r['local_test_loss']:.4f}{extra_str}")
            local_test_accs.append(r["local_test_accuracy"])
            summary_lines.append(
                f"  Worker {r['worker_id']}: local_test_accuracy={r['local_test_accuracy']:.4f}{extra_str}"
            )
        mean_lt = statistics.mean(local_test_accs)
        std_lt = statistics.stdev(local_test_accs) if len(local_test_accs) > 1 else 0.0
        print(f"\n  Mean local test accuracy: {mean_lt:.4f}  (std={std_lt:.4f})")
        summary_lines.append(f"\nMean local test accuracy: {mean_lt:.4f}  std={std_lt:.4f}")
        for label, vals_e in local_test_extra.items():
            if vals_e:
                mean_e = statistics.mean(vals_e)
                std_e = statistics.stdev(vals_e) if len(vals_e) > 1 else 0.0
                print(f"  Mean local test {label}: {mean_e:.4f}  (std={std_e:.4f})")
                summary_lines.append(f"Mean local test {label}: {mean_e:.4f}  std={std_e:.4f}")
        print("-" * 65)
        print()

    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------
    summary_fname = "summary_recomputed.txt" if is_archived_run else "summary.txt"
    summary_path = os.path.join(args.data_root, summary_fname)
    with open(summary_path, "w") as f:
        f.write(f"Experiment summary — {len(csv_files)} workers\n")
        f.write("=" * 60 + "\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write(f"\nTotal gossip messages sent: {total_sent}\n")
    if is_archived_run:
        print(f"Archived run — wrote a new file without touching existing ones: {summary_path}")
    else:
        print(f"Summary saved to:       {summary_path}")

    if args.plot:
        if is_archived_run:
            print("\nArchived run — skipping --plot (would overwrite the saved PNGs).")
        else:
            print()
            _plot_results(global_rows, worker_rows, args.data_root, has_timing, has_global_test)


if __name__ == "__main__":
    main()
