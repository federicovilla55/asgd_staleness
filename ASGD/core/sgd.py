from ..models import LinearNetModel
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

def sgd_training(X_train, y_train, num_epochs = 10000, criterion = nn.MSELoss(), batch_size = 32, lr = 0.01, tol=1e-8):

    # Create a linear model with dimention equal to the number of features
    # in the dataset
    model   = LinearNetModel(X_train.shape[1])

    # Train the model using standard SGD
    loader  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
           torch.from_numpy(X_train), torch.from_numpy(y_train)
        ),
        batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_epoch_loss = 0.0
        num_batches = 0

        # Iterate over the batches of training data
        for Xb, yb in loader:
            optimizer.zero_grad() # Reset the gradients
            out  = model(Xb) # Forward pass
            loss = criterion(out, yb.float()) # Compute the loss
            loss.backward() # Backward pass
            optimizer.step() # Update the model parameters
            total_epoch_loss += loss.item() # Accumulate the loss
            num_batches += 1
        
        avg_loss = total_epoch_loss / num_batches
        # Early stopping
        if avg_loss < tol:
            print(f"Stopping early at epoch {epoch} with avg loss {avg_loss:.6f} < tol={tol}")
            break

    return model


def sgd_training_dropout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_epochs: int = 10000,
    criterion: nn.Module = nn.MSELoss(),
    batch_size: int = 32,
    lr: float = 0.01,
    tol: float = 1e-8,
    dropout_p: float = 0.2
) -> LinearNetModel:

    # Create model
    model   = LinearNetModel(X_train.shape[1])
    # Create dropout layer
    dropout = nn.Dropout(p=dropout_p)

    # DataLoader
    loader  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
           torch.from_numpy(X_train).float(),
           torch.from_numpy(y_train).float()
        ),
        batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_epoch_loss = 0.0
        num_batches = 0

        for Xb, yb in loader:
            optimizer.zero_grad()
            # apply dropout to inputs
            Xb_dropped = dropout(Xb)
            out        = model(Xb_dropped)            # forward
            loss       = criterion(out, yb)          # compute loss
            loss.backward()                          # backward
            optimizer.step()                         # update
            total_epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_epoch_loss / num_batches
        if avg_loss < tol:
            print(f"Stopping early at epoch {epoch} with avg loss {avg_loss:.6f} < tol={tol}")
            break

    model.eval()  # disable dropout for any subsequent eval

    # Return the trained model
    return model


def sgd_training_l2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_epochs: int = 10000,
    criterion: nn.Module = nn.MSELoss(),
    batch_size: int = 32,
    lr: float = 0.01,
    tol: float = 1e-8,
    weight_decay: float = 1e-3
) -> LinearNetModel:
    model = LinearNetModel(X_train.shape[1])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float()
        ), batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    for epoch in range(num_epochs):
        total_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (total_loss/len(loader)) < tol:
            break
    return model


def sgd_training_noise(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_epochs: int = 10000,
    criterion: nn.Module = nn.MSELoss(),
    batch_size: int = 32,
    lr: float = 0.01,
    tol: float = 1e-8,
    noise_scale: float = 1e-3
) -> LinearNetModel:
    model = LinearNetModel(X_train.shape[1])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float()
        ), batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            # Gaussian gradient noise injection
            for p in model.parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * noise_scale
            optimizer.step()
            total_loss += loss.item()
        if (total_loss/len(loader)) < tol:
            break
    return model