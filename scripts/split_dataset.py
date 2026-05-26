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


def _stream_split_val_test(
    worker_map: dict[str, int],
    num_workers: int,
    worker_user_lists: list[list[str]],
    out_dirs_val: list[str],
    out_dirs_test: list[str],
) -> None:
    """
    Split LEAF test/ 50/50 per writer into val/ and test/ output files.

    For each writer, the first half of their samples goes to val/ (used for
    early stopping) and the second half goes to test/ (independent evaluation,
    never used for any training decision). Both output files contain the same
    writer IDs. The 20% LEAF test/ becomes 10% val + 10% test.
    """
    shard_files = sorted(glob.glob(os.path.join(SRC_DIR, "test", "*.json")))

    val_handles, test_handles = [], []
    for w in range(num_workers):
        for handles, out_dirs in [(val_handles, out_dirs_val), (test_handles, out_dirs_test)]:
            h = open(os.path.join(out_dirs[w], "data.json"), "w")
            h.write('{"users":')
            json.dump(worker_user_lists[w], h)
            h.write(',"user_data":{')
            handles.append(h)

    first_val = [True] * num_workers
    first_test = [True] * num_workers

    for idx, shard_path in enumerate(shard_files, 1):
        print(f"    shard {idx}/{len(shard_files)}: {os.path.basename(shard_path)}")
        with open(shard_path) as f:
            shard = json.load(f)

        for user in shard["users"]:
            w = worker_map[user]
            x = shard["user_data"][user]["x"]
            y = shard["user_data"][user]["y"]
            mid = max(1, len(x) // 2)

            if not first_val[w]:
                val_handles[w].write(",")
            val_handles[w].write(json.dumps(user) + ":" + json.dumps({"x": x[:mid], "y": y[:mid]}))
            first_val[w] = False

            if not first_test[w]:
                test_handles[w].write(",")
            test_handles[w].write(json.dumps(user) + ":" + json.dumps({"x": x[mid:], "y": y[mid:]}))
            first_test[w] = False

        del shard
        gc.collect()

    for h in val_handles + test_handles:
        h.write("}}")
        h.close()


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


def _process_split(
    src_split: str,
    dst_split: str,
    num_workers: int,
) -> None:
    """Process a single LEAF split directory → per-worker output directory."""
    src_dir = os.path.join(SRC_DIR, src_split)
    print(f"\n[{src_split} → {dst_split}]")

    all_users = _collect_user_ids(src_dir)
    chunk_size = len(all_users) // num_workers
    print(f"  {len(all_users)} writers total, ~{chunk_size} per worker")

    worker_map = _build_worker_map(all_users, num_workers)

    worker_user_lists: list[list[str]] = [[] for _ in range(num_workers)]
    for user in all_users:
        worker_user_lists[worker_map[user]].append(user)

    out_dirs = []
    for i in range(num_workers):
        d = os.path.join(DEST_ROOT, f"worker_{i}", dst_split)
        os.makedirs(d, exist_ok=True)
        out_dirs.append(d)

    _stream_split(src_split, worker_map, num_workers, worker_user_lists, out_dirs)


def main():
    cfg = _load_config()
    num_workers: int = cfg["network"]["num_workers"]
    use_test_set: bool = cfg["machine_learning"].get("use_test_set", False)

    if not os.path.isdir(SRC_DIR):
        print(f"ERROR: source dataset not found at {SRC_DIR}")
        print("Run scripts/download_femnist.py first.")
        raise SystemExit(1)

    # Remove stale per-worker directories from previous runs
    for entry in glob.glob(os.path.join(DEST_ROOT, "worker_*")):
        shutil.rmtree(entry)

    mode = "80/10/10 train/val/test" if use_test_set else "90/10 train/val"
    print(f"Splitting FEMNIST into {num_workers} partitions [{mode}] ...")

    # train/ → train/ is identical in both modes
    _process_split("train", "train", num_workers)

    if use_test_set:
        # LEAF test/ (20%) → split 50/50 per writer: val/ (10%) + test/ (10%).
        # val/ is used for early stopping only; test/ is evaluated once at the
        # end of training and never influences any training decision.
        print("\n[test → val + test (50/50 per writer)]")
        src_dir = os.path.join(SRC_DIR, "test")
        all_users = _collect_user_ids(src_dir)
        chunk_size = len(all_users) // num_workers
        print(f"  {len(all_users)} writers total, ~{chunk_size} per worker")

        worker_map = _build_worker_map(all_users, num_workers)
        worker_user_lists: list[list[str]] = [[] for _ in range(num_workers)]
        for user in all_users:
            worker_user_lists[worker_map[user]].append(user)

        out_dirs_val, out_dirs_test = [], []
        for i in range(num_workers):
            for out_dirs, dst in [(out_dirs_val, "val"), (out_dirs_test, "test")]:
                d = os.path.join(DEST_ROOT, f"worker_{i}", dst)
                os.makedirs(d, exist_ok=True)
                out_dirs.append(d)

        _stream_split_val_test(worker_map, num_workers, worker_user_lists, out_dirs_val, out_dirs_test)
    else:
        # LEAF uses "test/" but we rename it "val/" in per-worker directories
        # to reflect the actual usage: it is measured each round for early stopping,
        # not held out as a final test set. The LEAF source directory is unchanged.
        _process_split("test", "val", num_workers)

    print("\nDone. Per-worker partitions written to:")
    for i in range(num_workers):
        suffix = "{train,val,test}" if use_test_set else "{train,val}"
        print(f"  data/femnist/worker_{i}/{suffix}/")
    print("\nNext step: docker compose up --build")


if __name__ == "__main__":
    main()
