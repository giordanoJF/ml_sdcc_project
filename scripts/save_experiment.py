#!/usr/bin/env python3
"""
Archive the current experiment results for later comparison.

Run this after aggregate_metrics.py to save a snapshot of the current run.
Each snapshot includes the per-worker metrics, aggregated stats, test results
(if present), and the config.yaml used — so you can always trace back which
configuration produced which results.

Usage:
    python scripts/save_experiment.py <name>

    <name>: short label describing what was varied, e.g. lr_1e-3, fanout_2, baseline

Output:
    results/<timestamp>_<name>/
        config.yaml                    ← exact config used for this run
        global_metrics.csv             ← per-round aggregated stats
        summary.txt                    ← human-readable summary
        worker_0/metrics.csv           ← per-round per-worker metrics
        worker_0/local_test_result.json ← final test accuracy (if local_test_set: true)
        worker_1/...
        ...
"""
import argparse
import glob
import os
import shutil
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "femnist")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")


def save_docker_logs(dest):
    logs_dir = os.path.join(dest, "logs")
    os.makedirs(logs_dir)

    try:
        services = subprocess.check_output(
            ["docker", "compose", "config", "--services"],
            cwd=PROJECT_ROOT,
            text=True,
        ).splitlines()
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: could not list docker services: {e}")
        return []

    saved = []
    for svc in services:
        log_path = os.path.join(logs_dir, f"{svc}.log")
        try:
            logs = subprocess.check_output(
                ["docker", "compose", "logs", "--no-color", svc],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.STDOUT,
            )
            with open(log_path, "w") as f:
                f.write(logs)
            saved.append(f"logs/{svc}.log")
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: could not get logs for {svc}: {e}")

    return saved


def _append_termination_reason(dest: str) -> None:
    """Read the saved registry log and append the run termination reason to summary.txt."""
    registry_log = os.path.join(dest, "logs", "registry.log")
    summary_path = os.path.join(dest, "summary.txt")

    if not os.path.exists(registry_log):
        reason = "UNKNOWN — registry log not found"
    else:
        content = open(registry_log).read()
        if "RUN_TERMINATION: NORMAL" in content:
            reason = "NORMAL — all workers deregistered cleanly"
        elif "WATCHDOG_TIMEOUT" in content:
            reason = "WATCHDOG_TIMEOUT — registry forced shutdown after 60 min inactivity (ungraceful worker exits)"
        elif "Signal 15" in content or "Signal 2" in content:
            reason = "INTERRUPTED — manual stop (Ctrl+C / SIGTERM)"
        else:
            reason = "UNKNOWN — check logs/registry.log for details"

    line = f"\nRun termination: {reason}\n"
    print(f"  Run termination: {reason}")
    if os.path.exists(summary_path):
        with open(summary_path, "a") as f:
            f.write(line)


def main():
    parser = argparse.ArgumentParser(description="Archive current experiment results.")
    parser.add_argument("name", help="Short label for this run (e.g. lr_1e-3, fanout_2)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(RESULTS_ROOT, f"{timestamp}_{args.name}")
    os.makedirs(dest)

    copied = []

    # config.yaml — required to trace which configuration produced these results
    config_src = os.path.join(PROJECT_ROOT, "config.yaml")
    if os.path.exists(config_src):
        shutil.copy2(config_src, os.path.join(dest, "config.yaml"))
        copied.append("config.yaml")

    # Aggregated outputs from aggregate_metrics.py
    for fname in ("global_metrics.csv", "summary.txt",
                  "accuracy_over_rounds.png", "loss_over_rounds.png",
                  "phase_timing.png", "global_test_accuracy.png"):
        src = os.path.join(DATA_ROOT, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, fname))
            copied.append(fname)
        elif fname in ("global_metrics.csv", "summary.txt"):
            print(f"  WARNING: {fname} not found — run aggregate_metrics.py first")

    # Per-worker files
    for worker_dir in sorted(glob.glob(os.path.join(DATA_ROOT, "worker_*"))):
        worker_name = os.path.basename(worker_dir)
        worker_dest = os.path.join(dest, worker_name)
        os.makedirs(worker_dest)

        for fname in ("metrics.csv", "local_test_result.json", "model_best.pt"):
            src = os.path.join(worker_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(worker_dest, fname))
                copied.append(f"{worker_name}/{fname}")

    log_files = save_docker_logs(dest)
    copied.extend(log_files)

    _append_termination_reason(dest)

    print(f"Saved to: {dest}")
    print(f"Files archived: {len(copied)}")
    for f in copied:
        print(f"  {f}")

    # Clean working directory so the next run starts from a blank slate.
    cleaned = []
    for fname in ("global_metrics.csv", "summary.txt"):
        p = os.path.join(DATA_ROOT, fname)
        if os.path.exists(p):
            os.remove(p)
            cleaned.append(fname)
    for worker_dir in sorted(glob.glob(os.path.join(DATA_ROOT, "worker_*"))):
        for fname in ("metrics.csv", "local_test_result.json", "model_best.pt"):
            p = os.path.join(worker_dir, fname)
            if os.path.exists(p):
                os.remove(p)
                cleaned.append(os.path.join(os.path.basename(worker_dir), fname))


    for fname in ("accuracy_over_rounds.png", "loss_over_rounds.png",
                  "phase_timing.png", "global_test_accuracy.png"):
        p = os.path.join(DATA_ROOT, fname)
        if os.path.exists(p):
            os.remove(p)
            cleaned.append(fname)

    if cleaned:
        print(f"\nCleaned {len(cleaned)} file(s) from working directory (ready for next run):")
        for f in cleaned:
            print(f"  {f}")


if __name__ == "__main__":
    main()
