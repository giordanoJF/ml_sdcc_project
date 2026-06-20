#!/usr/bin/env python3
"""
Run the complete experimental plan (8 core runs).

Block A — H ablation  (N=3, fanout=1, lr=1e-3):
  ref   : H=500   ← reference for all comparisons
  h100  : H=100   ← more frequent gossip, less local drift
  h1000 : H=1000  ← less frequent gossip, more local drift
  → best_H determined automatically from Block A results

Block B — Fanout at N=3  (H=best_H):
  f2    : fanout=2  (=N-1, full broadcast for N=3)

Block C — Scalability  (H=best_H):
  n5_f1 : N=5, fanout=1          [risplit required]
  n5_fn : N=5, fanout=4  (=N-1)
  n8_f1 : N=8, fanout=1          [risplit required]
  n8_fn : N=8, fanout=7  (=N-1)

Usage:
    python scripts/run_grid.py              # full plan, skips already-done runs
    python scripts/run_grid.py --dry-run    # print plan without executing
    python scripts/run_grid.py --only ref h100 h1000
    python scripts/run_grid.py --from f2    # skip everything before f2
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RESULTS_ROOT = PROJECT_ROOT / "results"

# lr is fixed for all runs
LR_FIXED = 0.001

PLAN = [
    # ── Block A — H ablation, N=3, fanout=1 ──────────────────────────────
    {"label": "ref",   "block": "A", "n": 3, "fanout": 1, "h": 500},
    {"label": "h100",  "block": "A", "n": 3, "fanout": 1, "h": 100},
    {"label": "h1000", "block": "A", "n": 3, "fanout": 1, "h": 1000},
    # ── Block B — fanout at N=3, H=best_H ────────────────────────────────
    {"label": "f2",    "block": "B", "n": 3, "fanout": 2, "h": None},
    # ── Block C — scalability, H=best_H ──────────────────────────────────
    {"label": "n5_f1", "block": "C", "n": 5, "fanout": 1, "h": None},
    {"label": "n5_fn", "block": "C", "n": 5, "fanout": 4, "h": None},  # N-1
    {"label": "n8_f1", "block": "C", "n": 8, "fanout": 1, "h": None},
    {"label": "n8_fn", "block": "C", "n": 8, "fanout": 7, "h": None},  # N-1
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def already_done(label: str) -> bool:
    if not RESULTS_ROOT.is_dir():
        return False
    return any(e.endswith(f"_{label}") for e in os.listdir(RESULTS_ROOT))


def find_result_dir(label: str) -> Path | None:
    if not RESULTS_ROOT.is_dir():
        return None
    for e in sorted(os.listdir(RESULTS_ROOT)):
        if e.endswith(f"_{label}"):
            return RESULTS_ROOT / e
    return None


def read_mean_best_val(label: str) -> float | None:
    d = find_result_dir(label)
    if d is None:
        return None
    summary = d / "summary.txt"
    if not summary.exists():
        return None
    for line in summary.read_text().splitlines():
        m = re.search(r"mean_best_val_accuracy\s*=\s*([\d.]+)", line)
        if m:
            return float(m.group(1))
    return None


def determine_best_h() -> int:
    """Read Block A results and return the H with the highest mean_best_val_accuracy."""
    block_a = [(r["label"], r["h"]) for r in PLAN if r["block"] == "A"]
    print("\nDetermining best_H from Block A results:")
    scores: dict[str, tuple[int, float]] = {}
    for label, h in block_a:
        acc = read_mean_best_val(label)
        if acc is not None:
            scores[label] = (h, acc)
            print(f"  {label} (H={h}): mean_best_val_accuracy = {acc:.4f}")
        else:
            print(f"  {label}: no saved result found")
    if not scores:
        print("  WARNING: no Block A results — defaulting to H=500")
        return 500
    best_label = max(scores, key=lambda k: scores[k][1])
    best_h, best_acc = scores[best_label]
    print(f"  → best_H = {best_h}  (from '{best_label}', acc={best_acc:.4f})\n")
    return best_h


def update_config(n: int, fanout: int, h: int) -> None:
    text = CONFIG_PATH.read_text()
    text = re.sub(r"^(\s*num_workers:\s*)\d+",   rf"\g<1>{n}",      text, flags=re.MULTILINE)
    text = re.sub(r"^(\s*gossip_fanout:\s*)\d+",  rf"\g<1>{fanout}", text, flags=re.MULTILINE)
    text = re.sub(r"^(\s*inner_steps_H:\s*)\d+",  rf"\g<1>{h}",      text, flags=re.MULTILINE)
    CONFIG_PATH.write_text(text)


_COMPOSE_MAX_RETRIES = 5
_COMPOSE_RETRY_DELAY = 5  # seconds


def run_cmd(cmd: list, label: str, allow_fail: bool = False) -> int:
    print(f"[{label}] $ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run([str(c) for c in cmd], cwd=PROJECT_ROOT)
    if r.returncode != 0 and not allow_fail:
        print(f"[{label}] FAILED (exit {r.returncode})")
        sys.exit(r.returncode)
    return r.returncode


def save_crash_logs(label: str) -> None:
    """Save docker logs and config to results/ for a run that crashed or was interrupted."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = RESULTS_ROOT / f"{timestamp}_crashed_{label}"
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy2(str(CONFIG_PATH), str(dest / "config.yaml"))

    try:
        services = subprocess.run(
            ["docker", "compose", "config", "--services"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        ).stdout.splitlines()
        logs_dir = dest / "logs"
        logs_dir.mkdir()
        for svc in [s for s in services if s.strip()]:
            logs = subprocess.run(
                ["docker", "compose", "logs", "--no-color", svc],
                cwd=PROJECT_ROOT, capture_output=True, text=True,
            ).stdout
            (logs_dir / f"{svc}.log").write_text(logs)
        print(f"[{label}] Crash logs saved → {dest}")
    except Exception as e:
        print(f"[{label}] WARNING: could not save crash logs: {e}")


def run_compose_up(label: str) -> bool:
    """Run 'docker compose up --build' with automatic retry on failure.

    Returns True if the command eventually succeeded, False if all retries
    were exhausted (caller should skip the current run with `continue`).
    Docker pulls/builds are left to run to completion — we only retry after
    the process exits with a non-zero code.
    """
    cmd = ["docker", "compose", "up", "--build"]
    for attempt in range(1, _COMPOSE_MAX_RETRIES + 1):
        print(f"[{label}] $ {' '.join(cmd)}"
              + (f"  (attempt {attempt}/{_COMPOSE_MAX_RETRIES})" if attempt > 1 else ""))
        r = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if r.returncode == 0:
            return True
        print(f"[{label}] compose up failed (exit {r.returncode}), "
              f"attempt {attempt}/{_COMPOSE_MAX_RETRIES}")
        if attempt < _COMPOSE_MAX_RETRIES:
            print(f"[{label}] retrying in {_COMPOSE_RETRY_DELAY}s ...")
            time.sleep(_COMPOSE_RETRY_DELAY)
    print(f"[{label}] WARNING: compose up failed after {_COMPOSE_MAX_RETRIES} attempts — skipping run.")
    save_crash_logs(label)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FL experimental plan.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without executing anything")
    parser.add_argument("--only", nargs="+", metavar="LABEL",
                        help="Run only these labels (order from plan is preserved)")
    parser.add_argument("--from", dest="from_label", metavar="LABEL",
                        help="Skip all runs before this label")
    args = parser.parse_args()

    valid_labels = [r["label"] for r in PLAN]
    if args.from_label and args.from_label not in valid_labels:
        print(f"ERROR: unknown label '{args.from_label}'. Valid labels: {valid_labels}")
        sys.exit(1)
    if args.only:
        for lbl in args.only:
            if lbl not in valid_labels:
                print(f"ERROR: unknown label '{lbl}'. Valid labels: {valid_labels}")
                sys.exit(1)

    # Determine current_n from the most-recently-completed run in plan order.
    # This lets the script avoid unnecessary risplits when resuming.
    current_n: int | None = None
    for run in PLAN:
        if already_done(run["label"]):
            current_n = run["n"]

    # ── Print plan table ──────────────────────────────────────────────────
    print(f"\n{'#':<3} {'Blk':<5} {'Label':<10} {'N':<4} {'fanout':<7} {'H':<6}  Status")
    print("─" * 58)
    active_print = args.from_label is None
    for i, run in enumerate(PLAN, 1):
        if args.from_label and run["label"] == args.from_label:
            active_print = True
        h_str = str(run["h"]) if run["h"] is not None else "best_H"
        if not active_print:
            st = "skip (--from)"
        elif args.only and run["label"] not in args.only:
            st = "skip (--only)"
        elif already_done(run["label"]):
            st = "done ✓"
        else:
            st = "PENDING"
        print(f"{i:<3} {run['block']:<5} {run['label']:<10} {run['n']:<4} "
              f"{run['fanout']:<7} {h_str:<6}  {st}")
    print()

    if args.dry_run:
        return

    # ── Execute ───────────────────────────────────────────────────────────
    best_h: int | None = None
    active = args.from_label is None

    for run in PLAN:
        label = run["label"]
        n = run["n"]

        if args.from_label and label == args.from_label:
            active = True
        if not active:
            continue
        if args.only and label not in args.only:
            continue

        # Resolve H (None = use best_H from Block A)
        if run["h"] is None:
            if best_h is None:
                best_h = determine_best_h()
            h = best_h
        else:
            h = run["h"]

        fanout = run["fanout"]

        if already_done(label):
            print(f"[{label}] Already in results — skipping.")
            current_n = n
            continue

        print(f"\n{'=' * 58}")
        print(f"  Block {run['block']} › {label}  "
              f"(N={n}, fanout={fanout}, H={h}, lr={LR_FIXED})")
        print(f"{'=' * 58}\n")

        # 1. Update config.yaml
        update_config(n, fanout, h)

        # 2. Re-partition dataset if N changed (or first run — current_n is None)
        if n != current_n:
            print(f"[{label}] N changed ({current_n} → {n}): re-partitioning dataset ...")
            run_cmd([sys.executable, "scripts/split_dataset.py"],    label)
            run_cmd([sys.executable, "scripts/generate_compose.py"], label)
        current_n = n

        # 3. Stop any containers still running from the previous experiment
        run_cmd(["docker", "compose", "down"], label, allow_fail=True)

        # 4. Train (retries automatically on failure)
        if not run_compose_up(label):
            run_cmd(["docker", "compose", "down"], label, allow_fail=True)
            continue

        # 5. Analyse + archive (save_experiment.py also cleans the working dir)
        run_cmd([sys.executable, "scripts/aggregate_metrics.py", "--plot"], label)
        run_cmd([sys.executable, "scripts/save_experiment.py", label], label)

    print(f"\nAll runs complete.")


if __name__ == "__main__":
    main()
