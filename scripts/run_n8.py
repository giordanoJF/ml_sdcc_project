#!/usr/bin/env python3
"""
Quick test: run only the two N=8 configurations (n8_f1, n8_fn).

H defaults to 500 (same as the reference run). If Block A results are already
present in results/, best_H is read from them exactly as run_grid.py would.

Usage:
    python scripts/run_n8.py            # run both n8_f1 and n8_fn
    python scripts/run_n8.py --dry-run  # print plan only
"""
import argparse
import sys
from pathlib import Path

# Import helpers from run_grid (no side effects at import time)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_grid import (
    PROJECT_ROOT,
    LR_FIXED,
    already_done,
    determine_best_h,
    update_config,
    run_cmd,
    run_compose_up,
    save_crash_logs,
)

N8_RUNS = [
    {"label": "n8_f1", "n": 8, "fanout": 1},
    {"label": "n8_fn", "n": 8, "fanout": 7},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run N=8 FL experiments only.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--h", type=int, default=None,
                        help="Override H (default: read Block A results, fallback 500)")
    args = parser.parse_args()

    h = args.h if args.h is not None else determine_best_h()

    print(f"\n{'Label':<10} {'N':<4} {'fanout':<7} {'H':<6}  Status")
    print("─" * 40)
    for run in N8_RUNS:
        st = "done ✓" if already_done(run["label"]) else "PENDING"
        print(f"{run['label']:<10} {run['n']:<4} {run['fanout']:<7} {h:<6}  {st}")
    print()

    if args.dry_run:
        return

    for run in N8_RUNS:
        label  = run["label"]
        n      = run["n"]
        fanout = run["fanout"]

        if already_done(label):
            print(f"[{label}] Already in results — skipping.")
            continue

        print(f"\n{'=' * 50}")
        print(f"  {label}  (N={n}, fanout={fanout}, H={h}, lr={LR_FIXED})")
        print(f"{'=' * 50}\n")

        update_config(n, fanout, h)

        print(f"[{label}] Partitioning dataset for N={n} ...")
        run_cmd([sys.executable, "scripts/split_dataset.py"],    label)
        run_cmd([sys.executable, "scripts/generate_compose.py"], label)

        run_cmd(["docker", "compose", "down"], label, allow_fail=True)

        if not run_compose_up(label):
            run_cmd(["docker", "compose", "down"], label, allow_fail=True)
            continue

        run_cmd([sys.executable, "scripts/aggregate_metrics.py", "--plot"], label)
        run_cmd([sys.executable, "scripts/save_experiment.py", label],      label)

    print("\nDone.")


if __name__ == "__main__":
    main()
