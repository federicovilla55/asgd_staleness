"""A ultra‑simple builder that always returns the *complete* dataset.

Use this when you want every worker to look at identical data, exactly
like the `FullDataLoaderBuilder` from your notebook.
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .base import AbstractDataBuilder

class FullDataLoaderBuilder:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __call__(self, num_workers: int, batch_size: int, worker_id: int):
        ds     = TensorDataset(torch.from_numpy(self.X),
                               torch.from_numpy(self.y))
        loader = DataLoader(ds, batch_size=batch_size,
                            shuffle=True, drop_last=False)
        return loader, self.X.shape[1]