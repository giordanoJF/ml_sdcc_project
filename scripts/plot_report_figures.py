#!/usr/bin/env python3

import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "report_figures")

# (results/<dirname>, short label, environment)
RUNS = [
    ("20260620_195439_ref_run1",   "ref",       "local"),
    ("20260620_200020_h100_run2",  "h100",      "local"),
    ("20260620_200942_h1000_run3", "h1000",     "local"),
    ("20260620_201843_f2_run4",    "f2",        "local"),
    ("20260620_203734_n5_f1_run5", "n5_f1",     "local"),
    ("20260620_205044_n5_fn_run6", "n5_fn",     "local"),
    ("20260620_211325_n8_f1_run7", "n8_f1",     "local"),
    ("20260620_213754_n8_fn_run8", "n8_fn",     "local"),
    ("20260709_181238_AWS-RUN1-FIX", "n3_h500",   "aws"),
    ("20260709_181546_AWS-RUN2-FIX", "n3_h100",   "aws"),
    ("20260710_023241_AWS-RUN3-FIX", "n3_h1000",  "aws"),
    ("20260710_014136_AWS-RUN4-FIX", "n3_f2",     "aws"),
    ("20260710_101843_AWS-RUN5-FIX", "n5_f1",     "aws"),
    ("20260710_101126_AWS-RUN6-FIX", "n5_fn",     "aws"),
    ("20260710_143713_AWS-RUN7-FIX", "n8_f1",     "aws"),
    ("20260710_140006_AWS-RUN8-FIX", "n8_fn",     "aws"),
]

# Verified against core/model.py (FEMNISTModel, default hyperparameters):
# sum(p.numel() for p in model.parameters()) == 1_704_350
MODEL_PARAMS = 1_704_350
MODEL_MSG_MB = MODEL_PARAMS * 4 / 1e6  # float32 weights, decimal MB, no serialization overhead

ENV_STYLE = {"local": dict(color="#2b6cb0", marker="o"), "aws": dict(color="#c05621", marker="s")}
FANOUT_STYLE = {"min": dict(linestyle="-"), "max": dict(linestyle="--")}


def parse_summary(path):
    with open(path) as f:
        text = f.read()

    def grab(pattern, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    b_block = re.search(
        r"\[B\] At model_best\.pt round.*?mean=([\d.]+)\s+std=([\d.]+)\s+spread=([\d.]+)",
        text, re.S,
    )
    gt_mean, gt_std, gt_spread = (float(x) for x in b_block.groups()) if b_block else (None, None, None)

    return dict(
        val_acc_mean=grab(r"mean_best_val_accuracy = ([\d.]+)"),
        val_acc_std=grab(r"mean_best_val_accuracy = [\d.]+\s+std=([\d.]+)"),
        wall_clock_s=grab(r"System wall-clock total:\s+([\d.]+)s"),
        gossip_msgs=grab(r"Total gossip messages sent:\s+(\d+)", cast=int),
        l2_dist=grab(r"Mean pairwise L2 weight distance:\s+([\d.]+)"),
        global_test_mean=gt_mean,
        global_test_std=gt_std,
        global_test_spread=gt_spread,
    )


def parse_phase_timing(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "mean_phase_a_s" not in rows[0]:
        return dict(phase_a_s=None, phase_b_s=None, phase_c_s=None)
    n = len(rows)
    return dict(
        phase_a_s=sum(float(r["mean_phase_a_s"]) for r in rows) / n,
        phase_b_s=sum(float(r["mean_phase_b_s"]) for r in rows) / n,
        phase_c_s=sum(float(r["mean_phase_c_s"]) for r in rows) / n,
    )


def load_all_runs():
    data = []
    for dirname, label, env in RUNS:
        run_dir = os.path.join(RESULTS_DIR, dirname)
        with open(os.path.join(run_dir, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        row = dict(
            dirname=dirname, label=label, env=env,
            N=cfg["network"]["num_workers"],
            fanout=cfg["network"]["gossip_fanout"],
            H=cfg["federated_learning"]["inner_steps_H"],
            patience=cfg["federated_learning"]["early_stopping_patience"],
        )
        row.update(parse_summary(os.path.join(run_dir, "summary.txt")))
        row.update(parse_phase_timing(os.path.join(run_dir, "global_metrics.csv")))
        row["fanout_class"] = "min" if row["fanout"] == 1 else "max"
        data.append(row)
    return data


def scalability_subset(data):
    """H=1000 grid shared by both environments: fanout=1 vs fanout=N-1 at N=3,5,8."""
    return [r for r in data if r["H"] == 1000]


def fig_accuracy_vs_n(data, path):
    subset = scalability_subset(data)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharex=True)
    metrics = [("val_acc_mean", "val_acc_std", "Val. accuracy"),
               ("global_test_mean", "global_test_std", "Global-test accuracy")]
    for ax, (mkey, skey, title) in zip(axes, metrics):
        for env in ("local", "aws"):
            for fclass in ("min", "max"):
                pts = sorted((r["N"], r[mkey], r[skey]) for r in subset
                             if r["env"] == env and r["fanout_class"] == fclass)
                if not pts:
                    continue
                xs, ys, es = zip(*pts)
                style = {**ENV_STYLE[env], **FANOUT_STYLE[fclass]}
                fanout_desc = "k=1" if fclass == "min" else "k=N-1"
                ax.errorbar(xs, ys, yerr=es, capsize=3, label=f"{env} {fanout_desc}", **style)
        ax.set_xlabel("N (client count)")
        ax.set_ylabel(title)
        ax.set_xticks([3, 5, 8])
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig_communication_overhead(data, path):
    subset = scalability_subset(data)
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    width = 0.18
    ns = [3, 5, 8]
    offsets = {("local", "min"): -1.5, ("local", "max"): -0.5,
               ("aws", "min"): 0.5, ("aws", "max"): 1.5}
    for (env, fclass), off in offsets.items():
        ys = []
        for n in ns:
            match = [r for r in subset if r["N"] == n and r["env"] == env and r["fanout_class"] == fclass]
            ys.append(match[0]["gossip_msgs"] * MODEL_MSG_MB if match else 0)
        xs = [n + off * width for n in ns]
        fanout_desc = "k=1" if fclass == "min" else "k=N-1"
        color = ENV_STYLE[env]["color"]
        hatch = None if fclass == "min" else "//"
        ax.bar(xs, ys, width=width, label=f"{env} {fanout_desc}", color=color, alpha=0.85, hatch=hatch)
    ax.set_xticks(ns)
    ax.set_xlabel("N (client count)")
    ax.set_ylabel("Traffico gossip stimato (MB)")
    ax.legend(fontsize=6.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig_phase_timing(data, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharey=True)
    for ax, env in zip(axes, ("local", "aws")):
        rows = sorted((r for r in data if r["env"] == env), key=lambda r: (r["N"], r["fanout"]))
        labels = [r["label"] for r in rows]
        xs = range(len(rows))
        a = [r["phase_a_s"] for r in rows]
        b = [r["phase_b_s"] for r in rows]
        c = [r["phase_c_s"] for r in rows]
        ax.bar(xs, a, label="Fase A (merge+val)", color="#c05621")
        ax.bar(xs, b, bottom=a, label="Fase B (H step locali)", color="#2b6cb0")
        ab = [x + y for x, y in zip(a, b)]
        ax.bar(xs, c, bottom=ab, label="Fase C (gossip push)", color="#2f855a")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
        ax.set_title(env)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Durata media per round (s)")
    axes[1].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig_local_vs_aws(data, path):
    by_key = {}
    for r in data:
        key = (r["N"], r["fanout"], r["H"])
        by_key.setdefault(key, {})[r["env"]] = r
    pairs = sorted((k, v) for k, v in by_key.items() if "local" in v and "aws" in v)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    xs = range(len(pairs))
    labels = [f"N={k[0]},k={k[1]},H={k[2]}" for k, _ in pairs]

    ax = axes[0]
    local_wc = [v["local"]["wall_clock_s"] / 60 for _, v in pairs]
    aws_wc = [v["aws"]["wall_clock_s"] / 60 for _, v in pairs]
    width = 0.35
    ax.bar([x - width / 2 for x in xs], local_wc, width=width, label="locale", color=ENV_STYLE["local"]["color"])
    ax.bar([x + width / 2 for x in xs], aws_wc, width=width, label="AWS", color=ENV_STYLE["aws"]["color"])
    ax.set_yscale("log")
    ax.set_ylabel("Wall-clock di sistema (min, log)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    local_spread = [v["local"]["global_test_spread"] * 100 for _, v in pairs]
    aws_spread = [v["aws"]["global_test_spread"] * 100 for _, v in pairs]
    ax.bar([x - width / 2 for x in xs], local_spread, width=width, label="locale", color=ENV_STYLE["local"]["color"])
    ax.bar([x + width / 2 for x in xs], aws_spread, width=width, label="AWS", color=ENV_STYLE["aws"]["color"])
    ax.set_ylabel("Spread global-test (%, metrica B)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_all_runs()
    fig_accuracy_vs_n(data, os.path.join(OUTPUT_DIR, "accuracy_vs_n.pdf"))
    fig_communication_overhead(data, os.path.join(OUTPUT_DIR, "communication_overhead.pdf"))
    fig_phase_timing(data, os.path.join(OUTPUT_DIR, "phase_timing.pdf"))
    fig_local_vs_aws(data, os.path.join(OUTPUT_DIR, "local_vs_aws.pdf"))


if __name__ == "__main__":
    main()
