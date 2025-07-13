import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Tuple

def create_linear_dataset(n_samples=100,
                          n_features=110,
                          noise=0.0,
                          random_state=None):
    """
    Overparameterized linear regression dataset:
      - X sampled U(-3, 3)
      - y = X @ w_true + noise
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-3, 3, size=(n_samples, n_features))
    w_true = rng.randn(n_features)
    y = X.dot(w_true) + noise * rng.randn(n_samples)
    return X.astype(np.float32), y.astype(np.float32), w_true.astype(np.float32)

def create_poly_varied_dataset(n_samples=100,
                               n_features=110,
                               max_degree=4,
                               noise=0.0,
                               random_state=None):
    """
    Overparameterized nonlinear regression dataset:
      - X sampled U(-3, 3)
      - Each feature i raised to its own degree_i ∈ [1, max_degree]
      - y = sum_i w_true[i] * (X[:, i] ** degree_i) + noise
      
    Return: X_raw, y, degrees
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-3, 3, size=(n_samples, n_features))
    w_true = rng.randn(n_features)
    degrees = rng.randint(1, max_degree + 1, size=n_features)
    X_pow = np.stack([X[:, i] ** deg for i, deg in enumerate(degrees)], axis=1)
    y = X_pow.dot(w_true) + noise * rng.randn(n_samples)
    return X.astype(np.float32), y.astype(np.float32), degrees

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

def load_linear_data(n_samples=100,
                     n_features=110,
                     noise=0.0,
                     val_size=0.01,
                     test_size=0.2,
                     random_state=None):
    """
    Generate a linear overparam dataset and split it.
    Returns: X_train, y_train, X_val, y_val, X_test, y_test
    """
    X, y, w = create_linear_dataset(n_samples, n_features, noise, random_state)
    X_tr_lin, y_tr_lin, X_val_lin, y_val_lin, X_te_lin, y_te_lin = split_data(X, y, val_size, test_size, random_state)
    return X_tr_lin, y_tr_lin, X_val_lin, y_val_lin, X_te_lin, y_te_lin, w

def load_poly_varied_data(n_samples=100,
                          n_features=110,
                          max_degree=4,
                          noise=0.0,
                          val_size=0.2,
                          test_size=0.2,
                          random_state=42):
    """
    Generate a polynomial-varied dataset, split it, and also return degrees.
    Returns: (X_train, y_train, X_val, y_val, X_test, y_test, degrees)
    """
    X, y, degrees = create_poly_varied_dataset(
        n_samples, n_features, max_degree, noise, random_state)
    splits = split_data(X, y, val_size, test_size, random_state)
    return (*splits, degrees)

def create_linear_data_loader(num_workers,
                              batch_size,
                              worker_id,
                              n_samples=100,
                              n_features=110,
                              noise=0.0,
                              val_size=0.01,
                              test_size=0.2,
                              random_state=42):
    """
    Return a DataLoader for a shard of the linear training set.
    Also returns the input dimension.
    """
    X_train, y_train, _, _, _, _ = load_linear_data(
        n_samples, n_features, noise, val_size, test_size, random_state) #Give full dataset to each worker
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader, X_train.shape[1]

def create_poly_varied_data_loader(num_workers,
                                   batch_size,
                                   worker_id,
                                   n_samples=100,
                                   n_features=110,
                                   max_degree=4,
                                   noise=0.0,
                                   val_size=0.2,
                                   test_size=0.2,
                                   random_state=42):
    """
    Return a DataLoader for a shard of the poly-varied training set.
    Also returns the input dimension (same as n_features).
    """
    X_train, y_train, _, _, _, _, degrees = load_poly_varied_data(
        n_samples, n_features, max_degree, noise, val_size, test_size, random_state + worker_id)
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader, X_train.shape[1], degrees
