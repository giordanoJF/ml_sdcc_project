#!/usr/bin/env python3

import argparse
import glob
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)  # so `core.*` resolves when run as `python scripts/confusion_matrix.py`

from core.dataset import load_global_test
from core.model import FEMNISTModel
from core.trainer import compute_confusion_matrix, macro_prf1_from_confusion


def class_label(idx: int) -> str:
    """FEMNIST label convention: 0-9 digits, 10-35 uppercase A-Z, 36-61 lowercase a-z."""
    if idx < 10:
        return str(idx)
    elif idx < 36:
        return chr(ord("A") + idx - 10)
    else:
        return chr(ord("a") + idx - 36)


def main():
    parser = argparse.ArgumentParser(description="Confusion matrix over the global test set.")
    parser.add_argument(
        "--data-root",
        default=os.path.join(PROJECT_ROOT, "data", "femnist"),
        help="Root directory containing worker_*/model_best.pt",
    )
    parser.add_argument(
        "--global-test-dir",
        default=os.path.join(PROJECT_ROOT, "data", "femnist", "global_test"),
        help="Directory with the shared global test set (data.npy/labels.npy or data.json)",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--top-k", type=int, default=20, help="Number of confused pairs / worst classes to show")
    args = parser.parse_args()

    checkpoint_paths = sorted(glob.glob(os.path.join(args.data_root, "worker_*", "model_best.pt")))
    if not checkpoint_paths:
        print(f"No model_best.pt found under {args.data_root}/worker_*/")
        print("  (archived runs only have this if saved after the save_experiment.py checkpoint-archiving fix)")
        raise SystemExit(1)

    global_test_loader = load_global_test(args.global_test_dir, args.batch_size)
    if global_test_loader is None:
        print(f"No global test set found at {args.global_test_dir} — nothing to evaluate against.")
        raise SystemExit(1)
    print(f"Global test set: {len(global_test_loader.dataset)} samples, {len(checkpoint_paths)} checkpoint(s)\n")

    device = torch.device("cpu")
    total_confusion: torch.Tensor | None = None

    for path in checkpoint_paths:
        worker_name = os.path.basename(os.path.dirname(path))
        model = FEMNISTModel()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(device)

        confusion = compute_confusion_matrix(model, global_test_loader, device)
        precision, recall, f1 = macro_prf1_from_confusion(confusion)
        print(f"  {worker_name}: macro_precision={precision:.4f}  macro_recall={recall:.4f}  macro_f1={f1:.4f}")

        total_confusion = confusion.clone() if total_confusion is None else total_confusion + confusion

    num_classes = total_confusion.shape[0]
    precision, recall, f1 = macro_prf1_from_confusion(total_confusion)
    print(f"\nCombined ({len(checkpoint_paths)} worker(s) summed):")
    print(f"  macro_precision={precision:.4f}  macro_recall={recall:.4f}  macro_f1={f1:.4f}")

    # ---------------------------------------------------------------------------
    # Top-K most confused ordered pairs (true -> predicted, off-diagonal)
    # ---------------------------------------------------------------------------
    support = total_confusion.sum(dim=1)  # true count per class, for the % figure below
    off_diag = total_confusion.clone()
    off_diag.fill_diagonal_(0)
    flat_counts, flat_idx = off_diag.flatten().sort(descending=True)
    print(f"\nTop {args.top_k} most confused pairs (true -> predicted):")
    print("-" * 60)
    shown = 0
    for count, idx in zip(flat_counts.tolist(), flat_idx.tolist()):
        if count == 0 or shown >= args.top_k:
            break
        true_i, pred_j = idx // num_classes, idx % num_classes
        pct = 100.0 * count / support[true_i].item() if support[true_i] > 0 else 0.0
        print(f"  '{class_label(true_i)}' -> '{class_label(pred_j)}': {count:>6}  ({pct:.1f}% of true '{class_label(true_i)}' samples)")
        shown += 1
    if shown == 0:
        print("  (no misclassifications — perfect predictions)")

    # ---------------------------------------------------------------------------
    # Bottom-K classes by recall — the classes the model misses most
    # ---------------------------------------------------------------------------
    tp = total_confusion.diag().float()
    recall_per_class = torch.where(support > 0, tp / support.clamp(min=1).float(), torch.full_like(tp, float("nan")))
    present_classes = [(c, recall_per_class[c].item(), int(support[c].item()))
                        for c in range(num_classes) if support[c] > 0]
    present_classes.sort(key=lambda t: t[1])
    print(f"\nBottom {args.top_k} classes by recall (most-missed classes):")
    print("-" * 60)
    for c, rec, n in present_classes[: args.top_k]:
        print(f"  '{class_label(c)}': recall={rec:.4f}  (support={n})")

    # ---------------------------------------------------------------------------
    # Save full matrix for later inspection/plotting
    # ---------------------------------------------------------------------------
    out_path = os.path.join(args.data_root, "confusion_matrix.csv")
    labels = [class_label(c) for c in range(num_classes)]
    with open(out_path, "w") as f:
        f.write("true\\predicted," + ",".join(labels) + "\n")
        for i, row_label in enumerate(labels):
            f.write(row_label + "," + ",".join(str(v) for v in total_confusion[i].tolist()) + "\n")
    print(f"\nFull confusion matrix saved to: {out_path}")


if __name__ == "__main__":
    main()
