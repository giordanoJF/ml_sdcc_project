#!/usr/bin/env python3
"""
Download and preprocess the FEMNIST dataset from the LEAF repository.
Run this script ONCE before starting the system with docker compose.

Usage:
    python scripts/download_femnist.py [--sf 0.05]

    --sf: sampling fraction (default 0.05 = 5% of data; use 1.0 for the full dataset)

Output:
    data/femnist/data/train/*.json
    data/femnist/data/test/*.json
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

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

    t_start = time.time()
    print("=== FEMNIST download via LEAF ===")
    print(f"Sampling fraction: {args.sf * 100:.0f}%%")
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

    # Step 2 — Install Python dependencies required by LEAF's preprocessing scripts
    run([sys.executable, "-m", "pip", "install", "tensorflow-cpu", "Pillow", "numpy"])

    # Step 3 — Run LEAF's preprocessing script for FEMNIST with non-i.i.d. split
    #
    # preprocess.sh è lo script ufficiale di LEAF che costruisce il dataset FEMNIST
    # a partire dalle immagini raw di NIST. Esegue in sequenza:
    #   1. Download delle immagini originali NIST Special Database 19 (~900 MB)
    #   2. Segmentazione per scrittore: ogni immagine viene attribuita al suo autore
    #      (identificato da un codice univoco), creando una partizione naturalmente
    #      non-i.i.d. — ogni scrittore ha il proprio stile di scrittura.
    #   3. Campionamento (--sf): tiene solo la frazione specificata dei dati.
    #      Utile per esperimenti veloci (es. --sf 0.05 = 5% dei dati, ~100k immagini
    #      invece di ~800k), senza cambiare la struttura non-i.i.d. del dataset.
    #   4. Split per-scrittore e serializzazione in JSON (vedi sotto).
    #
    # Come funziona lo split e perché è per scrittore
    # ------------------------------------------------
    # Lo split produce due cartelle: train/ e test/. ATTENZIONE: entrambe contengono
    # GLI STESSI scrittori — non sono due insiemi di persone diverse. La differenza
    # è che per ogni scrittore, il 90% dei suoi campioni finisce in train/ e il 10%
    # in test/. Lo split è quindi a livello di campione DENTRO ogni scrittore,
    # non a livello di scrittore intero.
    #
    # L'alternativa sarebbe -t writer: alcuni scrittori interi in train/, altri in
    # test/. Questo è sbagliato per FL perché split_dataset.py assegna scrittori
    # a worker: un worker che riceve scrittori finiti interamente in test/ non avrebbe
    # dati di training; uno che li riceve interamente in train/ non potrebbe validare.
    # Con -t sample ogni worker ha garantito sia training che validation per tutti i
    # suoi scrittori assegnati.
    #
    # Nota sul naming: LEAF chiama questa cartella "test/", ma nel nostro sistema
    # viene usata come validation set (misurata ad ogni round in main_worker.py per
    # l'early stopping). Non esiste un test set separato tenuto fuori dal training.
    #
    # Output: train/*.json e test/*.json nel formato
    #   {"users": [...], "user_data": {"writer_id": {"x": [[784 float]], "y": [int]}}}
    # letti da split_dataset.py (distribuzione ai worker) e core/dataset.py (PyTorch).
    #
    # Usiamo preprocess.sh perché è l'unica modalità supportata da LEAF per la
    # suddivisione per scrittore (-s niid): farlo manualmente richiederebbe
    # replicare la pipeline di segmentazione OCR già implementata da LEAF.
    #
    # Flag usate:
    #   -s niid   → split non-i.i.d. per scrittore (obbligatorio per FL realistico)
    #   --sf      → sampling fraction (1.0 = dataset completo, 0.05 = debug veloce)
    #   -k 0      → nessun minimo di campioni per utente (include tutti gli scrittori)
    #   -t sample → split per campione dentro ogni scrittore (non per scrittore intero)
    #   --tf 0.9  → 90% dei campioni di ogni scrittore in train/, 10% in test/
    femnist_dir = os.path.join(LEAF_DIR, "data", "femnist")
    run(
        [
            "./preprocess.sh",
            "-s", "niid",         # non-i.i.d. split by writer
            "--sf", str(args.sf), # fraction of data to keep
            "-k", "0",            # no minimum samples per user
            "-t", "sample",       # split by sample (not by writer)
            "--tf", "0.9",        # 90% train / 10% test
        ],
        cwd=femnist_dir,
    )

    # Step 4 — Copy only train/ and test/ JSON files into the project's data directory.
    # LEAF produces several intermediate directories (raw images, pkl files, sampled
    # data, etc.) that are not needed by the workers and would waste gigabytes of disk.
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR)
    src = os.path.join(femnist_dir, "data")
    for split in ("train", "test"):
        shutil.copytree(os.path.join(src, split), os.path.join(DEST_DIR, split))

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
