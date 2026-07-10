#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

import yaml

# Project root is the parent directory of this script's directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAF_DIR = os.path.join(PROJECT_ROOT, "leaf")
DEST_DIR = os.path.join(PROJECT_ROOT, "data", "femnist", "data")


def run(cmd: list, **kwargs):
    """Print and execute a shell command, raising on non-zero exit."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Download FEMNIST from LEAF")
    parser.add_argument(
        "--sf",
        type=float,
        default=1.0,
        help="Sampling fraction: 1.0 = full dataset (default), 0.05 = 5%% (fast)",
    )
    args = parser.parse_args()

    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    local_test_set = cfg["machine_learning"].get("local_test_set", False)
    # --tf controls the per-writer train fraction in LEAF:
    #   local_test_set=false → 90% train, 10% LEAF-test (used as worker val)
    #   local_test_set=true  → 80% train, 20% LEAF-test (split 50/50 → val + local_test)
    # global_test_set does not affect the LEAF download: it carves out writers after
    # download, entirely within split_dataset.py, without changing per-writer sample counts.
    tf = 0.8 if local_test_set else 0.9

    t_start = time.time()
    print("=== FEMNIST download via LEAF ===")
    print(f"Sampling fraction: {args.sf * 100:.0f}%%")
    print(f"Split mode:        {'80/10/10 train/val/local_test (local_test_set=true)' if local_test_set else '90/10 train/val (local_test_set=false)'}")
    print(f"Output directory:  {DEST_DIR}\n")

    os.chdir(PROJECT_ROOT)

    # Step 1 — Clone the LEAF repository if not already present
    if not os.path.exists(LEAF_DIR):
        run(["git", "clone", "https://github.com/TalwalkarLab/leaf.git", LEAF_DIR])
    else:
        print("LEAF repository already present, skipping clone.")

    # Step 1b — Patch LEAF for Pillow >= 10.0 compatibility.
    # Image.ANTIALIAS was removed in Pillow 10.0 (it was an alias for Image.LANCZOS).
    # The two are identical: same Lanczos resampling filter, same pixel output.
    data_to_json = os.path.join(
        LEAF_DIR, "data", "femnist", "preprocess", "data_to_json.py"
    )
    with open(data_to_json) as f:
        src_text = f.read()
    if "Image.ANTIALIAS" in src_text:
        with open(data_to_json, "w") as f:
            f.write(src_text.replace("Image.ANTIALIAS", "Image.LANCZOS"))
        print("Patched data_to_json.py: Image.ANTIALIAS → Image.LANCZOS")

    # Step 1c — Patch get_data.sh to use Python's zipfile instead of `unzip`.
    get_data_sh = os.path.join(LEAF_DIR, "data", "femnist", "preprocess", "get_data.sh")
    with open(get_data_sh) as f:
        src_text = f.read()
    if "unzip" in src_text:
        patched_lines = []
        for line in src_text.splitlines():
            # Match `unzip [optional-flags] <file>` — LEAF uses no flags, but handle
            # them defensively in case the script changes.
            m = re.match(r"^(\s*)unzip(?:\s+-\w+)*\s+(\S+)", line)
            if m:
                indent, fname = m.group(1), m.group(2)
                patched_lines.append(
                    f"{indent}python3 -c \"import zipfile; zipfile.ZipFile('{fname}').extractall('.')\""
                )
            else:
                patched_lines.append(line)
        with open(get_data_sh, "w") as f:
            f.write("\n".join(patched_lines))
        print("Patched get_data.sh: unzip → python3 zipfile")

    # Step 2 — Install Python dependencies required by LEAF's preprocessing scripts
    run([sys.executable, "-m", "pip", "install", "tensorflow-cpu", "Pillow", "numpy"])

    # Step 3 — Run LEAF's preprocessing script for FEMNIST with non-i.i.d. split
    #
    # preprocess.sh scarica le immagini NIST , le attribuisce a ogni scrittore
    # e produce train/*.json e test/*.json. La struttura per scrittore è preservata
    # dentro i JSON (chiave user_data), non in cartelle separate.
    #
    # Le due flag principali controllano cose diverse:
    #   -s niid   → organizza i dati PER SCRITTORE (non-i.i.d.): ogni writer_id ha le
    #               proprie immagini raggruppate. Senza questa flag LEAF miscela tutto.
    #   -t sample → determina come fare lo split train/test: per singola immagine
    #               dentro ogni scrittore (90% train, 10% test per ciascuno).
    #               Con -t writer invece scrittori interi finirebbero solo in train
    #               o solo in test — alcuni worker di split_dataset.py riceverebbero
    #               scrittori senza dati di training o senza validation.
    #
    # Nota: LEAF chiama questa cartella "test/", ma noi la copiamo come "val/"
    # subito al passo 4 per riflettere l'uso reale (validation set per early
    # stopping), non un test set tenuto fuori dal training.
    #
    # Altre flag:
    #   --sf  → frazione del dataset da tenere (1.0 = completo, 0.05 = debug veloce)
    #   -k 0  → nessun minimo di campioni per scrittore (include tutti)
    #   --tf 0.9 → 90% train / 10% test per ogni scrittore
    femnist_dir = os.path.join(LEAF_DIR, "data", "femnist")
    run(
        [
            "./preprocess.sh",
            "-s", "niid",         # non-i.i.d. split by writer
            "--sf", str(args.sf), # fraction of data to keep
            "-k", "0",            # no minimum samples per user
            "-t", "sample",       # split by sample (not by writer)
            "--tf", str(tf),      # 0.9 → 90/10 (val only), 0.8 → 80/20 (val+test)
        ],
        cwd=femnist_dir,
    )

    # Step 4 — Copy only train/ and test/ JSON files into the project's data directory.
    # LEAF produces several intermediate directories (raw images, pkl files, sampled
    # data, etc.) that are not needed by the workers and would waste gigabytes of disk.
    # LEAF calls this split "test/" but we rename it to "val/" immediately since it
    # is used as validation data by the workers, not as a held-out test set.
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR)
    src = os.path.join(femnist_dir, "data")
    shutil.copytree(os.path.join(src, "train"), os.path.join(DEST_DIR, "train"))
    shutil.copytree(os.path.join(src, "test"), os.path.join(DEST_DIR, "val"))

    # Step 5 — Remove the LEAF repository: no longer needed after preprocessing.
    shutil.rmtree(LEAF_DIR)
    print("Removed LEAF repository (no longer needed).")

    elapsed = time.time() - t_start
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\nDataset ready at: {DEST_DIR}")
    print(f"Total time: {minutes}m {seconds}s")
    print("Next step: python scripts/split_dataset.py")


if __name__ == "__main__":
    main()
