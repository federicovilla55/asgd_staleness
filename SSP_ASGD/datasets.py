# datasets.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_adult_data(random_state=42):
    """
    Load the UCI Adult dataset from OpenML and preprocess it.
    
    :param random_state: Random state for reproducibility.
    :type random_state: int
    :return: Tuple of training features, training labels, test features, test labels.
    :rtype: tuple
    """
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    df    = adult.frame
    X = df.drop("class", axis=1)
    y = (df["class"] == ">50K").astype(int)
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X)
    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(np.int64)
    return train_test_split(
        X_np, 
        y_np, 
        test_size=0.2, 
        random_state=random_state, 
        stratify=y_np
    )

def create_adult_dataset(num_workers, batch_size, worker_id, random_state=42):
    """
    Create a DataLoader for the Adult dataset. This dataoader creates a shard of the dataset
    specific for the provided worker id. The dataset is split into num_workers shards.

    :param num_workers: Number of workers.
    :type num_workers: int
    :param batch_size: Batch size for the DataLoader.
    :type batch_size: int
    :param worker_id: ID of the worker.
    :type worker_id: int
    :param random_state: Random state for reproducibility.
    :type random_state: int
    :return: DataLoader for the worker and the number of features in the dataset.
    :rtype: tuple
    """
    X_train, _, y_train, _ = load_adult_data(random_state + worker_id)
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader, X_train.shape[1]

# To Do: test on the 3D polynomial dataset.

