#!/usr/bin/env python3
"""
Split the downloaded FEMNIST dataset into per-worker partitions.

Run this script AFTER download_femnist.py and BEFORE docker compose up.
Re-run whenever num_workers, local_test_set, or global_test_set in config.yaml changes.

Usage:
    python scripts/split_dataset.py

Input:
    data/femnist/data/train/*.json
    data/femnist/data/val/*.json

Output (local_test_set: false, global_test_set: false — default):
    data/femnist/worker_0/train/data.json
    data/femnist/worker_0/val/data.json
    data/femnist/worker_1/...

Output (local_test_set: true):
    data/femnist/worker_0/train/data.json
    data/femnist/worker_0/val/data.json
    data/femnist/worker_0/local_test/data.json
    data/femnist/worker_1/...

Output (global_test_set: true, additional):
    data/femnist/global_test/data.json   ← ALL samples of the carved-out writers
                                            (LEAF train + LEAF test merged), never in any worker dir

Memory strategy: two-pass streaming with immediate disk writes.
  Pass 1 — read only writer IDs (no pixel data) to build the global
            ordered list and compute the writer→worker mapping.
  Pass 2 — open all output files simultaneously; stream shards one
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
    return all_users


def _build_worker_map(all_users: list[str], num_workers: int) -> dict[str, int]:
    # Map each writer ID to its worker index using contiguous slicing.
    # min() ensures the last writer always goes to worker (num_workers-1)
    # even when len(all_users) is not exactly divisible by num_workers.
    # max(1, ...) guards against ZeroDivisionError when num_workers > len(all_users).
    chunk_size = max(1, len(all_users) // num_workers)
    return {user: min(i // chunk_size, num_workers - 1) for i, user in enumerate(all_users)}


def _stream_split(
    split: str,
    worker_map: dict[str, int],
    num_workers: int,
    worker_user_lists: list[list[str]],
    out_dirs: list[str],
    global_buffer: dict | None = None,
) -> None:
    """
    Pass 2: stream through all shards once, writing each writer's data
    immediately to the correct worker output file.

    Writers not in worker_map (global test writers) are accumulated into
    global_buffer if provided — their train samples are merged with their
    test samples later in _stream_test_split, then written to disk once.
    """
    shard_files = sorted(glob.glob(os.path.join(SRC_DIR, split, "*.json")))

    handles = []
    for w in range(num_workers):
        h = open(os.path.join(out_dirs[w], "data.json"), "w")
        h.write('{"users":')
        json.dump(worker_user_lists[w], h)
        h.write(',"user_data":{')
        handles.append(h)

    first_entry = [True] * num_workers

    for idx, shard_path in enumerate(shard_files, 1):
        print(f"    shard {idx}/{len(shard_files)}: {os.path.basename(shard_path)}")
        with open(shard_path) as f:
            shard = json.load(f)

        for user in shard["users"]:
            if user not in worker_map:
                if global_buffer is not None:
                    data = shard["user_data"][user]
                    if user not in global_buffer:
                        global_buffer[user] = {"x": list(data["x"]), "y": list(data["y"])}
                    else:
                        global_buffer[user]["x"].extend(data["x"])
                        global_buffer[user]["y"].extend(data["y"])
                continue
            w = worker_map[user]
            if not first_entry[w]:
                handles[w].write(",")
            handles[w].write(json.dumps(user))
            handles[w].write(":")
            handles[w].write(json.dumps(shard["user_data"][user]))
            first_entry[w] = False

        del shard
        gc.collect()

    for h in handles:
        h.write("}}")
        h.close()


def _stream_test_split(
    worker_map: dict[str, int],
    num_workers: int,
    worker_user_lists: list[list[str]],
    global_users_set: set[str],
    global_buffer: dict | None,
    out_dirs_val: list[str],
    out_dirs_local_test: list[str] | None,
) -> None:
    """
    Stream through the LEAF test/ split and route each writer to the correct output:

    - Writers in global_users_set → global_buffer (test samples merged with train
      samples already accumulated in _stream_split; written to disk once in main).
    - Remaining writers:
        - If out_dirs_local_test is None  → all samples go to worker val/
        - If out_dirs_local_test is given → samples split 50/50 per writer:
            first half → worker val/ (early stopping)
            second half → worker local_test/ (independent final evaluation)
    """
    shard_files = sorted(glob.glob(os.path.join(SRC_DIR, "val", "*.json")))

    # Open per-worker val files
    val_handles = []
    for w in range(num_workers):
        h = open(os.path.join(out_dirs_val[w], "data.json"), "w")
        h.write('{"users":')
        json.dump(worker_user_lists[w], h)
        h.write(',"user_data":{')
        val_handles.append(h)

    # Open per-worker local_test files (only when local_test_set: true)
    local_test_handles = []
    if out_dirs_local_test is not None:
        for w in range(num_workers):
            h = open(os.path.join(out_dirs_local_test[w], "data.json"), "w")
            h.write('{"users":')
            json.dump(worker_user_lists[w], h)
            h.write(',"user_data":{')
            local_test_handles.append(h)

    first_val = [True] * num_workers
    first_local_test = [True] * num_workers

    for idx, shard_path in enumerate(shard_files, 1):
        print(f"    shard {idx}/{len(shard_files)}: {os.path.basename(shard_path)}")
        with open(shard_path) as f:
            shard = json.load(f)

        for user in shard["users"]:
            x = shard["user_data"][user]["x"]
            y = shard["user_data"][user]["y"]

            if user in global_users_set:
                # Merge test samples into the buffer alongside the train samples
                # already accumulated in _stream_split. Written to disk once in main().
                if global_buffer is not None:
                    if user not in global_buffer:
                        global_buffer[user] = {"x": list(x), "y": list(y)}
                    else:
                        global_buffer[user]["x"].extend(x)
                        global_buffer[user]["y"].extend(y)
            else:
                # → per-worker val (and optionally local_test)
                w = worker_map[user]
                if out_dirs_local_test is not None:
                    # 50/50 per-writer split: val ← first half, local_test ← second half
                    mid = max(1, len(x) // 2)
                    if not first_val[w]:
                        val_handles[w].write(",")
                    val_handles[w].write(json.dumps(user) + ":" + json.dumps({"x": x[:mid], "y": y[:mid]}))
                    first_val[w] = False

                    if not first_local_test[w]:
                        local_test_handles[w].write(",")
                    local_test_handles[w].write(json.dumps(user) + ":" + json.dumps({"x": x[mid:], "y": y[mid:]}))
                    first_local_test[w] = False
                else:
                    # All samples → val
                    if not first_val[w]:
                        val_handles[w].write(",")
                    val_handles[w].write(json.dumps(user) + ":" + json.dumps({"x": x, "y": y}))
                    first_val[w] = False

        del shard
        gc.collect()

    for h in val_handles:
        h.write("}}")
        h.close()
    for h in local_test_handles:
        h.write("}}")
        h.close()


def _process_split(
    src_split: str,
    dst_split: str,
    num_workers: int,
    exclude: set[str] | None = None,
    global_buffer: dict | None = None,
) -> None:
    """Process a single LEAF split directory → per-worker output directory."""
    src_dir = os.path.join(SRC_DIR, src_split)
    print(f"\n[{src_split} → {dst_split}]")

    all_users = _collect_user_ids(src_dir)
    if exclude:
        all_users = [u for u in all_users if u not in exclude]
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

    _stream_split(src_split, worker_map, num_workers, worker_user_lists, out_dirs,
                  global_buffer=global_buffer)


def main():
    cfg = _load_config()
    num_workers: int = cfg["network"]["num_workers"]
    ml_cfg = cfg["machine_learning"]
    local_test_set: bool = ml_cfg.get("local_test_set", False)
    global_test_set: bool = ml_cfg.get("global_test_set", False)
    global_test_fraction: float = ml_cfg.get("global_test_fraction", 0.10)

    if not os.path.isdir(SRC_DIR):
        print(f"ERROR: source dataset not found at {SRC_DIR}")
        print("Run scripts/download_femnist.py first.")
        raise SystemExit(1)

    # Remove stale per-worker directories and global test dir from previous runs
    for entry in glob.glob(os.path.join(DEST_ROOT, "worker_*")):
        shutil.rmtree(entry)
    global_test_dir_host = os.path.join(DEST_ROOT, "global_test")
    if os.path.isdir(global_test_dir_host):
        shutil.rmtree(global_test_dir_host)

    parts = []
    if local_test_set:
        parts.append("80/10/10 train/val/local_test per worker")
    else:
        parts.append("90/10 train/val per worker")
    if global_test_set:
        parts.append(f"global_test ({int(global_test_fraction * 100)}% of test writers)")
    print(f"Splitting FEMNIST into {num_workers} partitions [{', '.join(parts)}] ...")

    # Determine global test writers before processing train/ so they can be excluded.
    # Global test writers must never appear in any worker's train/ — otherwise they
    # would not be truly unseen writers and the global test set would be meaningless.
    src_test_dir = os.path.join(SRC_DIR, "val")
    all_test_users = _collect_user_ids(src_test_dir)

    if global_test_set:
        n_global = max(1, int(len(all_test_users) * global_test_fraction))
        global_user_list = all_test_users[:n_global]
        worker_test_users = all_test_users[n_global:]
    else:
        global_user_list = []
        worker_test_users = all_test_users

    # global_buffer accumulates ALL samples (train + test) for global test writers.
    # Using a buffer avoids duplicate JSON keys that would arise from writing train
    # and test data for the same writer in two separate streaming passes.
    global_users_set = set(global_user_list)
    global_buffer: dict | None = {} if global_test_set and global_user_list else None

    # train/ → per-worker train/; global test writers' train samples → global_buffer
    _process_split("train", "train", num_workers, exclude=global_users_set,
                   global_buffer=global_buffer)

    # val/ → worker val/ (+ local_test/); global test writers' val samples → global_buffer
    print("\n[val → worker val/" + (" + local_test (50/50 per writer)" if local_test_set else "") +
          (" + global_test" if global_test_set else "") + "]")

    if global_test_set:
        print(f"  {len(all_test_users)} test writers total: "
              f"{n_global} → global_test, {len(worker_test_users)} → workers")
    else:
        print(f"  {len(all_test_users)} test writers → workers")

    chunk_size = len(worker_test_users) // num_workers
    print(f"  ~{chunk_size} val writers per worker")

    worker_map = _build_worker_map(worker_test_users, num_workers)
    worker_user_lists: list[list[str]] = [[] for _ in range(num_workers)]
    for user in worker_test_users:
        worker_user_lists[worker_map[user]].append(user)

    out_dirs_val = []
    out_dirs_local_test = [] if local_test_set else None
    for i in range(num_workers):
        val_d = os.path.join(DEST_ROOT, f"worker_{i}", "val")
        os.makedirs(val_d, exist_ok=True)
        out_dirs_val.append(val_d)
        if local_test_set:
            lt_d = os.path.join(DEST_ROOT, f"worker_{i}", "local_test")
            os.makedirs(lt_d, exist_ok=True)
            out_dirs_local_test.append(lt_d)

    _stream_test_split(
        worker_map=worker_map,
        num_workers=num_workers,
        worker_user_lists=worker_user_lists,
        global_users_set=global_users_set,
        global_buffer=global_buffer,
        out_dirs_val=out_dirs_val,
        out_dirs_local_test=out_dirs_local_test,
    )

    # Write global_test/data.json once, with train+test samples merged per writer.
    if global_buffer is not None:
        os.makedirs(global_test_dir_host, exist_ok=True)
        with open(os.path.join(global_test_dir_host, "data.json"), "w") as f:
            json.dump({"users": global_user_list, "user_data": global_buffer}, f)

    # Remove the LEAF source shards — no longer needed once worker dirs are written.
    shutil.rmtree(SRC_DIR)

    print("\nDone. Per-worker partitions written to:")
    for i in range(num_workers):
        dirs = "{train,val"
        if local_test_set:
            dirs += ",local_test"
        dirs += "}"
        print(f"  data/femnist/worker_{i}/{dirs}/")
    if global_test_set:
        print(f"  data/femnist/global_test/  ← shared, never assigned to any worker")
    print("\nNext step: python scripts/generate_compose.py && docker compose up --build")


if __name__ == "__main__":
    main()
