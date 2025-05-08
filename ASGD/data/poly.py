"""Non-linear “poly-varied” synthetic regression builder."""

from __future__ import annotations
import numpy as np
from .base import AbstractDataBuilder

class PolyVariedBuilder(AbstractDataBuilder):
    def __init__(
        self,
        num_workers: int,
        n_samples: int = 100,
        n_features: int = 110,
        max_degree: int = 4,
        noise: float = 0.0,
        shard: bool = False,
        seed: int | None = None,
    ):
        super().__init__(num_workers, seed)
        self.n_samples = n_samples
        self.n_features = n_features
        self.max_degree = max_degree
        self.noise = noise
        self.shard = shard

        self._X, self._y, self.degrees = self._build_dataset()

    def _create_poly_varied_dataset(
        n_samples: int,
        n_features: int,
        max_degree: int,
        noise: float,
        rng: np.random.RandomState,
    ):
        X = rng.uniform(-3, 3, size=(n_samples, n_features))
        w = rng.randn(n_features)
        degrees = rng.randint(1, max_degree + 1, size=n_features)
        X_pow = np.stack([X[:, i] ** d for i, d in enumerate(degrees)], axis=1)
        y = X_pow @ w + noise * rng.randn(n_samples)
        return X.astype(np.float32), y.astype(np.float32), degrees


    

    def _build_dataset(self):
        rng = np.random.RandomState(self.seed)
        return self._create_poly_varied_dataset(
            self.n_samples,
            self.n_features,
            self.max_degree,
            self.noise,
            rng,
        )

    def _slice_for_worker(self, X, y, worker_id):
        if not self.shard:
            return X, y
        return X[worker_id :: self.num_workers], y[worker_id :: self.num_workers]
