"""
FEMNIST dataset loading (LEAF format).

Each worker's partition is pre-split by scripts/split_dataset.py and
mounted into the container at the path specified by data_dir in config.yaml.
This module simply loads whatever train/ and val/ (and optionally local_test/)
JSON files are present in that directory — no runtime splitting logic.

Data model at this stage:
  - One writer  = one real person with a unique handwriting style.
  - One sample  = one 28x28 grayscale image, stored as a flat vector of
                  784 floats in [0, 1].
  - One label   = integer in [0, 61]:  0-9 digits, 10-35 uppercase A-Z,
                  36-61 lowercase a-z.
"""
import glob
import json
import os

import torch
from torch.utils.data import DataLoader, Dataset


class FEMNISTDataset(Dataset):
    """
    PyTorch Dataset wrapping LEAF FEMNIST JSON data.
    Images: 28x28 grayscale (flattened to 784 floats in LEAF format).
    Classes: 62 (digits 0-9 and letters a-z, A-Z).
    """

    def __init__(self, x_data: list, y_data: list):
        # x_data: list of flat vectors [784 floats] — one per image.
        # Reshape to (N, 1, 28, 28): batch × channels × height × width,
        # the format expected by Conv2d (1 channel = grayscale).
        self.x = torch.tensor(x_data, dtype=torch.float32).view(-1, 1, 28, 28)
        self.y = torch.tensor(y_data, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _read_json_shards(directory: str) -> tuple[list, dict]:
    """
    Read all LEAF JSON shards from a directory.
    Returns (all_users, user_data_map) merged across all files.

    After this call:
      all_users : ["f1967_21", ...]  — ordered list of writer IDs in this partition
      user_data : {"f1967_21": {"x": [[784 floats], ...], "y": [int, ...]}, ...}
    """
    all_users: list = []
    user_data: dict = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as f:
            shard = json.load(f)
        all_users.extend(shard["users"])
        user_data.update(shard["user_data"])
    return all_users, user_data


def _collect_samples(users: list, data_map: dict) -> tuple[list, list]:
    # Flatten all writers' samples into two parallel lists.
    # After this, writer identity is lost — we only keep (image, label) pairs.
    x, y = [], []
    for user in users:
        x.extend(data_map[user]["x"])
        y.extend(data_map[user]["y"])
    return x, y


def load_partition(
    data_dir: str, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader | None, int]:
    """
    Load the train/val(/local_test) split from the pre-split worker directory.

    The directory must contain train/ and val/ subdirectories produced by
    scripts/split_dataset.py. If a local_test/ subdirectory is also present
    (local_test_set: true in config.yaml), a local_test_loader is returned;
    otherwise it is None.

    local_test_loader, when present, contains samples from the same writers
    as this worker's val/ set but held out from gradient updates. It provides
    an unbiased estimate of generalisation on the worker's own writer population.
    Evaluate it only once at the end of training — never for early stopping.

    Returns:
        train_loader, val_loader, local_test_loader (or None), num_train_samples
    """
    train_users, train_data = _read_json_shards(os.path.join(data_dir, "train"))
    val_users, val_data = _read_json_shards(os.path.join(data_dir, "val"))

    train_x, train_y = _collect_samples(train_users, train_data)
    val_x, val_y = _collect_samples(val_users, val_data)

    train_loader = DataLoader(
        FEMNISTDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        FEMNISTDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False,
    )

    local_test_loader = None
    local_test_dir = os.path.join(data_dir, "local_test")
    if os.path.isdir(local_test_dir):
        lt_users, lt_data = _read_json_shards(local_test_dir)
        lt_x, lt_y = _collect_samples(lt_users, lt_data)
        local_test_loader = DataLoader(
            FEMNISTDataset(lt_x, lt_y),
            batch_size=batch_size,
            shuffle=False,
        )

    return train_loader, val_loader, local_test_loader, len(train_x)


def load_global_test(global_test_dir: str, batch_size: int) -> DataLoader | None:
    """
    Load the shared global test set from global_test_dir.

    The global test set contains writers carved out before any per-worker split
    by scripts/split_dataset.py (global_test_set: true in config.yaml). These
    writers never appear in any worker's train/, val/, or local_test/ directories.

    Evaluating all workers on this identical set at each round reveals functional
    convergence: if workers that started from random initializations and trained
    on disjoint non-i.i.d. partitions reach the same accuracy on unseen writers,
    the gossip protocol has driven them to the same functional solution — not just
    nearby parameters (measured by L2 weight distance).

    Returns None if global_test_dir does not exist (global_test_set: false).
    """
    if not os.path.isdir(global_test_dir):
        return None
    users, data = _read_json_shards(global_test_dir)
    x, y = _collect_samples(users, data)
    return DataLoader(
        FEMNISTDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
    )
