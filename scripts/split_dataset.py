#!/usr/bin/env python3

import gc
import glob
import json
import os
import shutil

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR   = os.path.join(PROJECT_ROOT, "data", "femnist", "data")
DEST_ROOT = os.path.join(PROJECT_ROOT, "data", "femnist")


def _collect_user_ids(directory: str) -> list[str]:
    """Pass 1: read only 'users' keys from all shards — no pixel data loaded."""
    users: list[str] = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as f:
            users.extend(json.load(f)["users"])
    return users


def _build_worker_map(users: list[str], num_workers: int) -> dict[str, int]:
    chunk = max(1, len(users) // num_workers)
    return {u: min(i // chunk, num_workers - 1) for i, u in enumerate(users)}


def _make_user_lists(
    worker_writers: list[str],
    worker_map: dict[str, int],
    src_set: set[str],
    num_workers: int,
) -> list[list[str]]:
    """Per-worker user lists restricted to writers present in src_set.

    dataset.py iterates this list and looks up each writer in user_data —
    so it must contain only writers that actually have data in that split.
    """
    lists: list[list[str]] = [[] for _ in range(num_workers)]
    for u in worker_writers:
        if u in src_set:
            lists[worker_map[u]].append(u)
    return lists


def _open_worker_files(
    out_dirs: list[str], user_lists: list[list[str]]
) -> tuple[list, list[bool]]:
    """Open per-worker JSON files and write the header (users + opening of user_data)."""
    handles, first = [], []
    for i, d in enumerate(out_dirs):
        os.makedirs(d, exist_ok=True)
        h = open(os.path.join(d, "data.json"), "w")
        h.write('{"users":')
        json.dump(user_lists[i], h)
        h.write(',"user_data":{')
        handles.append(h)
        first.append(True)
    return handles, first


def _close_worker_files(handles: list) -> None:
    for h in handles:
        h.write("}}")
        h.close()


def _save_npy(directory: str) -> None:
    """Read the partition JSON from directory and save data.npy + labels.npy alongside it."""
    with open(os.path.join(directory, "data.json")) as f:
        shard = json.load(f)
    users = shard["users"]
    data  = shard["user_data"]
    total = sum(len(data[u]["y"]) for u in users)
    x = np.empty((total, 784), dtype=np.float32)
    y = np.empty(total, dtype=np.int64)
    idx = 0
    for user in users:
        ud = data[user]
        n  = len(ud["y"])
        x[idx : idx + n] = ud["x"]
        y[idx : idx + n] = ud["y"]
        idx += n
    del data, shard
    np.save(os.path.join(directory, "data.npy"),   x)
    np.save(os.path.join(directory, "labels.npy"), y)


def _stream(
    src_dir: str,
    worker_map: dict[str, int],
    global_users_set: set[str],
    worker_handles: list,
    first_entry: list[bool],
    global_buffer: dict | None,
    local_test_handles: list | None = None,
    first_local_test: list[bool] | None = None,
) -> None:
    """Stream all shards in src_dir, routing each writer to the correct output.

    Global writers go to global_buffer (train + val samples merged).
    Worker writers go to their assigned worker file.
    When local_test_handles is given (val/ pass, local_test_set=true), val data
    is split 50/50 per writer: first half → val, second half → local_test.
    With download_femnist.py using --tf 0.8 this gives 80/10/10 train/val/local_test.
    """
    shard_paths = sorted(glob.glob(os.path.join(src_dir, "*.json")))
    for idx, path in enumerate(shard_paths, 1):
        print(f"  shard {idx}/{len(shard_paths)}: {os.path.basename(path)}")
        with open(path) as f:
            shard = json.load(f)

        for user in shard["users"]:
            data = shard["user_data"][user]

            if user in global_users_set:
                if global_buffer is not None:
                    buf = global_buffer.setdefault(user, {"x": [], "y": []})
                    buf["x"].extend(data["x"])
                    buf["y"].extend(data["y"])
                continue

            if user not in worker_map:
                continue

            w = worker_map[user]

            if local_test_handles is not None:
                mid = max(1, len(data["x"]) // 2)
                val_data = {"x": data["x"][:mid], "y": data["y"][:mid]}
                lt_data  = {"x": data["x"][mid:], "y": data["y"][mid:]}

                if not first_entry[w]:
                    worker_handles[w].write(",")
                worker_handles[w].write(json.dumps(user) + ":" + json.dumps(val_data))
                first_entry[w] = False

                if not first_local_test[w]:
                    local_test_handles[w].write(",")
                local_test_handles[w].write(json.dumps(user) + ":" + json.dumps(lt_data))
                first_local_test[w] = False
            else:
                if not first_entry[w]:
                    worker_handles[w].write(",")
                worker_handles[w].write(json.dumps(user) + ":" + json.dumps(data))
                first_entry[w] = False

        del shard
        gc.collect()


def main():
    cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config.yaml")))
    num_workers      = cfg["network"]["num_workers"]
    ml               = cfg["machine_learning"]
    local_test_set   = ml.get("local_test_set", False)
    global_test_set  = ml.get("global_test_set", False)
    global_test_frac = ml.get("global_test_fraction", 0.10)

    if not os.path.isdir(SRC_DIR):
        print(f"ERROR: source dataset not found at {SRC_DIR}")
        print("Run scripts/download_femnist.py first.")
        raise SystemExit(1)

    for entry in glob.glob(os.path.join(DEST_ROOT, "worker_*")):
        shutil.rmtree(entry)
    global_test_dir = os.path.join(DEST_ROOT, "global_test")
    if os.path.isdir(global_test_dir):
        shutil.rmtree(global_test_dir)

    #Pass 1: single source of truth
    #
    # Sorted union of train/ and val/ writers: fully deterministic regardless of
    # LEAF's internal shard ordering. No assumption on which split comes first
    # or how LEAF shuffled writers between train/ and test/.
    train_users_set = set(_collect_user_ids(os.path.join(SRC_DIR, "train")))
    val_users_set   = set(_collect_user_ids(os.path.join(SRC_DIR, "val")))
    all_writers     = sorted(train_users_set | val_users_set)

    n_global = max(1, int(len(all_writers) * global_test_frac)) if global_test_set else 0
    global_users_set = set(all_writers[:n_global])
    worker_writers   = [u for u in all_writers if u not in global_users_set]

    # Single worker_map used for train/, val/, and local_test/
    worker_map = _build_worker_map(worker_writers, num_workers)

    # Per-split user lists for JSON headers (restricted to writers present in that split)
    train_user_lists = _make_user_lists(worker_writers, worker_map, train_users_set, num_workers)
    val_user_lists   = _make_user_lists(worker_writers, worker_map, val_users_set,   num_workers)

    parts = ["80/10/10 train/val/local_test" if local_test_set else "90/10 train/val"]
    if global_test_set:
        parts.append(f"global_test ({int(global_test_frac * 100)}% of writers)")
    print(f"Splitting FEMNIST into {num_workers} partitions [{', '.join(parts)}] ...")
    print(f"  {len(all_writers)} writers total | {n_global} → global_test | {len(worker_writers)} → workers (~{len(worker_writers) // num_workers} per worker)")

    global_buffer: dict | None = {} if global_test_set else None

    # Pass 2a: train/
    train_dirs = [os.path.join(DEST_ROOT, f"worker_{i}", "train") for i in range(num_workers)]
    train_handles, first_train = _open_worker_files(train_dirs, train_user_lists)

    print("\n[train/]")
    _stream(os.path.join(SRC_DIR, "train"), worker_map, global_users_set,
            train_handles, first_train, global_buffer)
    _close_worker_files(train_handles)

    # Pass 2b: val/ (+ optional local_test/)
    val_dirs = [os.path.join(DEST_ROOT, f"worker_{i}", "val") for i in range(num_workers)]
    val_handles, first_val = _open_worker_files(val_dirs, val_user_lists)

    lt_handles, first_lt = None, None
    if local_test_set:
        lt_dirs = [os.path.join(DEST_ROOT, f"worker_{i}", "local_test") for i in range(num_workers)]
        lt_handles, first_lt = _open_worker_files(lt_dirs, val_user_lists)

    print("\n[val/]")
    _stream(os.path.join(SRC_DIR, "val"), worker_map, global_users_set,
            val_handles, first_val, global_buffer,
            local_test_handles=lt_handles, first_local_test=first_lt)
    _close_worker_files(val_handles)
    if lt_handles:
        _close_worker_files(lt_handles)

    # Write global_test/
    if global_buffer is not None:
        os.makedirs(global_test_dir, exist_ok=True)
        with open(os.path.join(global_test_dir, "data.json"), "w") as f:
            json.dump({"users": all_writers[:n_global], "user_data": global_buffer}, f)

        # Pre-convert to numpy binary so containers can load via mmap (read-only mount).
        # All workers share the same physical pages → ~400 MB total instead of N × 3 GB.
        total = sum(len(v["y"]) for v in global_buffer.values())
        gt_x = np.empty((total, 784), dtype=np.float32)
        gt_y = np.empty(total, dtype=np.int64)
        idx = 0
        for user in all_writers[:n_global]:
            if user in global_buffer:
                ud = global_buffer[user]
                n = len(ud["y"])
                gt_x[idx : idx + n] = ud["x"]
                gt_y[idx : idx + n] = ud["y"]
                idx += n
        np.save(os.path.join(global_test_dir, "data.npy"),   gt_x[:idx])
        np.save(os.path.join(global_test_dir, "labels.npy"), gt_y[:idx])

    # Pass 3: pre-generate .npy for fast loading in worker containers
    # JSON-parsed Python float objects cost ~24 bytes/float (vs 4 as float32).
    # For N=8 workers, JSON peak RAM was ~1.9 GB. Binary .npy files let dataset.py skip JSON entirely: load
    # peak drops to the array size alone.
    npy_splits = ["train", "val"] + (["local_test"] if local_test_set else [])
    print(f"\n[pre-generating .npy ({', '.join(npy_splits)})]")
    for i in range(num_workers):
        for split in npy_splits:
            split_dir = os.path.join(DEST_ROOT, f"worker_{i}", split)
            print(f"  worker_{i}/{split} ...", end=" ", flush=True)
            _save_npy(split_dir)
            print("done")

    print("\nDone. Per-worker partitions written to:")
    for i in range(num_workers):
        splits = "train, val" + (", local_test" if local_test_set else "")
        print(f"  data/femnist/worker_{i}/  [{splits}]")
    if global_test_set:
        print(f"  data/femnist/global_test/")
    print("\nNext step: python scripts/generate_compose.py && docker compose up --build")


if __name__ == "__main__":
    main()
