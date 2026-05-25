"""
FEMNIST dataset loading (LEAF format).

Each worker's partition is pre-split by scripts/split_dataset.py and
mounted into the container at the path specified by data_dir in config.yaml.
This module simply loads whatever train/ and test/ JSON files are present
in that directory — no runtime splitting logic.

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


def load_partition(
    data_dir: str, batch_size: int
) -> tuple[DataLoader, DataLoader, int]:
    """
    Load the train/validation split from the pre-split worker directory.

    The directory is expected to contain train/ and test/ subdirectories
    with the JSON files produced by scripts/split_dataset.py.

    Returns:
        train_loader, val_loader, num_train_samples
    """
    train_users, train_data = _read_json_shards(os.path.join(data_dir, "train"))
    val_users, val_data = _read_json_shards(os.path.join(data_dir, "val"))

    def collect_samples(users: list, data_map: dict) -> tuple[list, list]:
        # Flatten all writers' samples into two parallel lists.
        # After this, writer identity is lost — we only keep (image, label) pairs.
        # This is intentional: the model trains on samples, not on writer identity.
        x, y = [], []
        for user in users:
            x.extend(data_map[user]["x"])   # list of [784 floats]
            y.extend(data_map[user]["y"])   # list of int labels
        return x, y

    train_x, train_y = collect_samples(train_users, train_data)
    val_x, val_y = collect_samples(val_users, val_data)

    # At this point:
    #   train_x : list of N flat vectors, each [784 floats] — all training images
    #   train_y : list of N integers in [0, 61]             — corresponding labels
    # FEMNISTDataset will convert these to tensors and reshape x to (N, 1, 28, 28).

    train_loader = DataLoader(
        FEMNISTDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # avoid partial batches that could skew gradient estimates
    )
    val_loader = DataLoader(
        FEMNISTDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, len(train_x)
