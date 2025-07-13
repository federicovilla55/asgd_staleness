"""Shared abstract class for dataset builders."""

from __future__ import annotations
import abc
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class AbstractDataBuilder(abc.ABC):
    def __init__(self, num_workers: int, seed: int | None = None):
        self.num_workers = num_workers
        self.seed = seed

    @abc.abstractmethod
    def _build_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def _slice_for_worker(
        self, X: np.ndarray, y: np.ndarray, worker_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        return X, y

    def __call__(
        self, batch_size: int, worker_id: int
    ) -> Tuple[DataLoader, int]:
        X, y = self._build_dataset()
        X_w, y_w = self._slice_for_worker(X, y, worker_id)

        ds = TensorDataset(torch.from_numpy(X_w), torch.from_numpy(y_w))
        loader = DataLoader(ds, batch_size=batch_size,
                            shuffle=True, drop_last=False)
        return loader, X.shape[1]

    def split_data(X, y, val_size=0.0, test_size=0.2, random_state=None):
        """
        Splits (X, y) into train/val/test.
        - train: (1 - val_size - test_size)
        - val: val_size
        - test: test_size
        """
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        val_rel = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_rel, random_state=random_state)
        return X_train, y_train, X_val, y_val, X_test, y_test


def _train_val_test_split(
    X: np.ndarray, y: np.ndarray,
    val_size: float, test_size: float, seed: int | None):

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test