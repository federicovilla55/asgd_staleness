"""Synthetic over-parameterised linear regression dataset builder."""

from __future__ import annotations
import numpy as np
from .base import AbstractDataBuilder

class LinearRegressionBuilder(AbstractDataBuilder):
    def __init__(
        self,
        num_workers: int,
        n_samples: int = 100,
        n_features: int = 110,
        noise: float = 0.0,
        shard: bool = False,
        seed: int | None = None,
    ):
        super().__init__(num_workers, seed)
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        self.shard = shard

        self._X, self._y = self._build_dataset()

    def _create_linear_dataset(
        n_samples: int,
        n_features: int,
        noise: float,
        rng: np.random.RandomState,
    ):
        X = rng.uniform(-3, 3, size=(n_samples, n_features)).astype(np.float32)
        w = rng.randn(n_features)
        y = (X @ w + noise * rng.randn(n_samples)).astype(np.float32)
        return X, y

    def _build_dataset(self):
        rng = np.random.RandomState(self.seed)
        return self._create_linear_dataset(
            self.n_samples, self.n_features, self.noise, rng
        )

    def _slice_for_worker(self, X, y, worker_id):
        if not self.shard:
            return X, y
        return X[worker_id :: self.num_workers], y[worker_id :: self.num_workers]
