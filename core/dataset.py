
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class FEMNISTDataset(Dataset):


    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x = torch.from_numpy(np.ascontiguousarray(x_data, dtype=np.float32)).view(-1, 1, 28, 28)
        self.y = torch.from_numpy(np.ascontiguousarray(y_data, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _read_json_shards(directory: str) -> tuple[list, dict]:

    all_users: list = []
    user_data: dict = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as f:
            shard = json.load(f)
        all_users.extend(shard["users"])
        user_data.update(shard["user_data"])
    return all_users, user_data


def _collect_samples(users: list, data_map: dict) -> tuple[np.ndarray, np.ndarray]:

    total = sum(len(data_map[u]["y"]) for u in users)
    x = np.empty((total, 784), dtype=np.float32)
    y = np.empty(total, dtype=np.int64)
    idx = 0
    for user in users:
        ud = data_map[user]
        n = len(ud["y"])
        x[idx : idx + n] = ud["x"]
        y[idx : idx + n] = ud["y"]
        idx += n
    return x, y


def _load_split(directory: str) -> tuple[np.ndarray, np.ndarray]:

    npy_x = os.path.join(directory, "data.npy")
    npy_y = os.path.join(directory, "labels.npy")
    if os.path.exists(npy_x) and os.path.exists(npy_y):
        return np.load(npy_x), np.load(npy_y)
    users, data = _read_json_shards(directory)
    x, y = _collect_samples(users, data)
    del data
    return x, y


def load_partition(
    data_dir: str, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader | None, int]:

    train_x, train_y = _load_split(os.path.join(data_dir, "train"))
    val_x, val_y     = _load_split(os.path.join(data_dir, "val"))

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
        lt_x, lt_y = _load_split(local_test_dir)
        local_test_loader = DataLoader(
            FEMNISTDataset(lt_x, lt_y),
            batch_size=batch_size,
            shuffle=False,
        )

    return train_loader, val_loader, local_test_loader, int(train_x.shape[0])


def load_global_test(global_test_dir: str, batch_size: int) -> DataLoader | None:

    if not os.path.isdir(global_test_dir):
        return None

    npy_x = os.path.join(global_test_dir, "data.npy")
    npy_y = os.path.join(global_test_dir, "labels.npy")

    if os.path.exists(npy_x) and os.path.exists(npy_y):

        x = np.load(npy_x, mmap_mode="c")
        y = np.load(npy_y, mmap_mode="c")
    else:
        users, data = _read_json_shards(global_test_dir)
        x, y = _collect_samples(users, data)
        del data

    return DataLoader(
        FEMNISTDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
    )
