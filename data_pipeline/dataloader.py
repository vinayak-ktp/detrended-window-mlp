import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

FEATURE_COLS = [
    "wds_scaled",
    "sp",
    "t2m",
    "sin_hour",
    "cos_hour",
    "sin_month",
    "cos_month",
    "sin_dayofyear",
    "cos_dayofyear",
]

TARGET_COL = "wds_scaled"


class WindDataset(Dataset):
    def __init__(self, data_path, seq_len=48, forecast_len=1, stride=1):
        df = pd.read_csv(data_path)
        self.features = df[FEATURE_COLS].values.astype(np.float32)
        self.targets = df[TARGET_COL].values.astype(np.float32)
        self.seq_len = seq_len
        self.forecast_len = forecast_len
        self.indices = range(0, len(self.features) - self.seq_len - self.forecast_len + 1, stride)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        seq_end = start + self.seq_len
        x = self.features[start : seq_end]
        y = self.targets[seq_end : seq_end + self.forecast_len]
        return torch.from_numpy(x), torch.from_numpy(y)


def get_dataloader(
    data_path,
    seq_len,
    forecast_len,
    batch_size,
    stride=1,
    shuffle=False,
    pin_memory=False
):
    dataset = WindDataset(
        data_path=data_path,
        seq_len=seq_len,
        forecast_len=forecast_len,
        stride=stride
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )
    return dataset, dataloader
