#!/usr/bin/env python3
"""
Split the downloaded FEMNIST dataset into per-worker partitions.

Run this script AFTER download_femnist.py and BEFORE docker compose up.
Re-run whenever num_workers in config.yaml changes.

Usage:
    python scripts/split_dataset.py

Input:
    data/femnist/data/train/*.json
    data/femnist/data/test/*.json

Output:
    data/femnist/worker_0/train/data.json
    data/femnist/worker_0/test/data.json
    data/femnist/worker_1/train/data.json
    ...

Memory strategy: two-pass streaming with immediate disk writes.
  Pass 1 — read only writer IDs (no pixel data) to build the global
            ordered list and compute the writer→worker mapping.
  Pass 2 — open all worker output files simultaneously; stream shards one
            at a time; write each writer's data directly to the correct
            output file the moment it is read, then discard the shard.
            Peak RAM = one shard (~1-2 GB as Python objects) regardless
            of dataset size or number of workers.
"""
import gc
import glob
import json
import os
import shutil

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "data", "femnist", "data")
DEST_ROOT = os.path.join(PROJECT_ROOT, "data", "femnist")


def _load_config() -> dict:
    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


def _collect_user_ids(directory: str) -> list[str]:
    # Pass 1: read only the "users" key — no pixel data loaded into memory.
    # Produces the global ordered list of writer IDs used to compute worker slices.
    all_users: list[str] = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as f:
            shard = json.load(f)
        all_users.extend(shard["users"])
        # Only string IDs are kept; pixel data is never loaded in this pass.
    return all_users


def _build_worker_map(all_users: list[str], num_workers: int) -> dict[str, int]:
    # Map each writer ID to its worker index using contiguous slicing.
    # min() ensures the last writer always goes to worker (num_workers-1)
    # even when len(all_users) is not exactly divisible by num_workers.
    chunk_size = len(all_users) // num_workers
    return {user: min(i // chunk_size, num_workers - 1) for i, user in enumerate(all_users)}


def _stream_split(
    split: str,
    worker_map: dict[str, int],
    num_workers: int,
    worker_user_lists: list[list[str]],
    out_dirs: list[str],
) -> None:
    """
    Pass 2: stream through all shards once, writing each writer's data
    immediately to the correct worker output file.

    All worker output files are kept open simultaneously so that a single
    pass over the shards is enough — no re-reading needed.
    The JSON is built manually (not via json.dump of the full dict) so that
    we never hold the complete dataset in memory at any point.
    """
    shard_files = sorted(glob.glob(os.path.join(SRC_DIR, split, "*.json")))

    # Open one output file per worker and write the JSON opening:
    #   {"users": [...list of string IDs, already known from pass 1...],
    #    "user_data": {
    handles = []
    for w in range(num_workers):
        h = open(os.path.join(out_dirs[w], "data.json"), "w")
        h.write('{"users":')
        # worker_user_lists[w] is just a list of string IDs — negligible memory
        json.dump(worker_user_lists[w], h)
        h.write(',"user_data":{')
        handles.append(h)

    first_entry = [True] * num_workers  # tracks whether to prepend a comma

    for idx, shard_path in enumerate(shard_files, 1):
        print(f"    shard {idx}/{len(shard_files)}: {os.path.basename(shard_path)}")
        with open(shard_path) as f:
            # One shard in memory at a time (~1-2 GB as Python objects).
            # After the loop body, shard goes out of scope and GC reclaims it.
            shard = json.load(f)

        for user in shard["users"]:
            w = worker_map[user]
            if not first_entry[w]:
                handles[w].write(",")
            # Write this writer's entry directly to the output file:
            #   "writer_id": {"x": [[784 floats], ...], "y": [int, ...]}
            handles[w].write(json.dumps(user))
            handles[w].write(":")
            handles[w].write(json.dumps(shard["user_data"][user]))
            first_entry[w] = False

        del shard       # explicit delete to help the GC reclaim RAM promptly
        gc.collect()

    # Close the JSON structure and all file handles
    for h in handles:
        h.write("}}")
        h.close()


def main():
    cfg = _load_config()
    num_workers: int = cfg["network"]["num_workers"]

    if not os.path.isdir(SRC_DIR):
        print(f"ERROR: source dataset not found at {SRC_DIR}")
        print("Run scripts/download_femnist.py first.")
        raise SystemExit(1)

    # Remove stale per-worker directories from previous runs
    for entry in glob.glob(os.path.join(DEST_ROOT, "worker_*")):
        shutil.rmtree(entry)

    print(f"Splitting FEMNIST into {num_workers} partitions ...")

    # LEAF uses "train/" and "test/" as directory names, but the "test/" split
    # is used as a validation set in our system (measured each round for early
    # stopping). We rename it to "val/" in the per-worker directories so the
    # code reflects the actual usage. The LEAF source directories are unchanged.
    split_mapping = {"train": "train", "test": "val"}

    for src_split, dst_split in split_mapping.items():
        src_dir = os.path.join(SRC_DIR, src_split)
        print(f"\n[{src_split} → {dst_split}]")

        # Pass 1: collect writer IDs only (no pixel data)
        all_users = _collect_user_ids(src_dir)
        chunk_size = len(all_users) // num_workers
        print(f"  {len(all_users)} writers total, ~{chunk_size} per worker")

        worker_map = _build_worker_map(all_users, num_workers)

        # Group writer IDs by worker (strings only — negligible memory)
        worker_user_lists: list[list[str]] = [[] for _ in range(num_workers)]
        for user in all_users:
            worker_user_lists[worker_map[user]].append(user)

        # Create output directories
        out_dirs = []
        for i in range(num_workers):
            d = os.path.join(DEST_ROOT, f"worker_{i}", dst_split)
            os.makedirs(d, exist_ok=True)
            out_dirs.append(d)

        # Pass 2: stream shards, write output files immediately
        _stream_split(src_split, worker_map, num_workers, worker_user_lists, out_dirs)

    print("\nDone. Per-worker partitions written to:")
    for i in range(num_workers):
        print(f"  data/femnist/worker_{i}/")
    print("\nNext step: docker compose up --build")


if __name__ == "__main__":
    main()
