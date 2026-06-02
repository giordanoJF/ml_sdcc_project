#!/usr/bin/env python3
"""
Automated grid search runner for Experiment 1.

Iterates over all (learning_rate, inner_steps_H) combinations, updates
config.yaml for each, runs docker compose, aggregates metrics, and saves
results. Already-completed runs (detected by existing results/ folder) are
skipped automatically.

Usage:
    python scripts/run_grid.py              # full 3x3 grid
    python scripts/run_grid.py --dry-run    # print plan without executing
    python scripts/run_grid.py --skip lr_1e-3_h_500  # skip specific runs
"""
import argparse
import os
import re
import subprocess
import sys
from itertools import product

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

# Experiment 1 grid — 3x3
LEARNING_RATES = [1e-4, 1e-3, 5e-3]
INNER_STEPS = [100, 500, 1000]


def run_label(lr: float, h: int) -> str:
    lr_str = f"{lr:.0e}".replace("-0", "-").replace("+0", "")
    return f"lr_{lr_str}_h_{h}"


def already_done(label: str) -> bool:
    """Check if a results folder for this label already exists."""
    for entry in os.listdir(RESULTS_ROOT):
        if entry.endswith(f"_{label}"):
            return True
    return False


def update_config(lr: float, h: int) -> None:
    with open(CONFIG_PATH) as f:
        text = f.read()
    # Use regex substitution to preserve all comments and formatting
    text = re.sub(r"^(\s*learning_rate:\s*)[\d.e+-]+", f"\\g<1>{lr}", text, flags=re.MULTILINE)
    text = re.sub(r"^(\s*inner_steps_H:\s*)\d+", f"\\g<1>{h}", text, flags=re.MULTILINE)
    with open(CONFIG_PATH, "w") as f:
        f.write(text)


def run_cmd(cmd: list[str], label: str) -> None:
    print(f"\n[{label}] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[{label}] FAILED with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--skip", nargs="*", default=[], metavar="LABEL",
                        help="Labels to skip, e.g. lr_1e-3_h_500")
    args = parser.parse_args()

    grid = [(lr, h) for lr, h in product(LEARNING_RATES, INNER_STEPS)]

    print(f"Grid search — {len(grid)} runs total")
    print(f"{'Run':<4} {'Label':<20} {'Status'}")
    print("-" * 45)
    to_run = []
    for i, (lr, h) in enumerate(grid, 1):
        label = run_label(lr, h)
        if label in args.skip:
            status = "SKIP (--skip)"
        elif already_done(label):
            status = "SKIP (exists)"
        else:
            status = "PENDING"
            to_run.append((lr, h, label))
        print(f"{i:<4} {label:<20} {status}")

    print(f"\n{len(to_run)} run(s) to execute.")

    if args.dry_run or not to_run:
        return

    for lr, h, label in to_run:
        print(f"\n{'='*50}")
        print(f"Starting: {label}  (lr={lr}, H={h})")
        print("=" * 50)

        update_config(lr, h)
        run_cmd(["docker", "compose", "up", "--build"], label)
        run_cmd(["python3.13", "scripts/aggregate_metrics.py", "--plot"], label)
        run_cmd(["python3.13", "scripts/save_experiment.py", label], label)

    print(f"\nAll {len(to_run)} run(s) complete.")


if __name__ == "__main__":
    main()
