import time, random
import numpy as np
import torch
import torch.nn as nn
from ..models import LinearNetModel
from scipy.stats import kurtosis
import argparse

def exp_delay(num_workers: int, scale: float = 1e-4) -> None:
    mean_stale = num_workers - 1
    time.sleep(np.random.exponential(mean_stale * scale))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def sgd_training(X_train, y_train, num_epochs = 10000, criterion = nn.MSELoss(), batch_size = 10, lr = 0.01, tol=1e-8):
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
    count_updates = 0


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
            count_updates += 1
        
        avg_loss = total_epoch_loss / num_batches
        # Early stopping
        if avg_loss < tol:
            print(f"SGD used {count_updates} updates")
            print(f"Stopping early at epoch {epoch} with avg loss {avg_loss:.6f} < tol={tol}")
            break

    return model

def build_model(theta: list[torch.Tensor], model, input_dim: int) -> nn.Module:
    """
    Build a model instance from the provided parameters.

    :param theta: List of model parameters.
    :type theta: list[torch.Tensor]
    :param model_cls: Model class to be instantiated.
    :type model_cls: Callable[[int], nn.Module]
    :param input_dim: Input dimension of the model.
    :type input_dim: int
    :return: Model instance with the provided parameters.
    :rtype: nn.Module
    """
    model = model(input_dim)
    with torch.no_grad():
        for param, trained_param in zip(model.parameters(), theta):
            param.copy_(trained_param)
    return model

def evaluate_model(name:str, model: nn.Module, X_eval: np.ndarray, y_eval: np.ndarray) -> float:
    """
    Evaluate the model on the provided evaluation dataset.

    :param model: Model instance to be evaluated.
    :type model: nn.Module
    :param X_eval: Evaluation dataset features.
    :type X_eval: np.ndarray
    :param y_eval: Evaluation dataset labels.
    :type y_eval: np.ndarray
    :return: Accuracy of the model on the evaluation dataset.
    :rtype: float
    :raises ValueError: If the model is not in evaluation mode.
    """
    # ensure in eval mode
    model.eval()

    # Move data into torch tensors
    X_tensor = torch.from_numpy(X_eval).float()
    y_tensor = torch.from_numpy(y_eval).float()

    # For MSE, use the built‑in loss (mean reduction by default)
    criterion = nn.MSELoss()

    with torch.no_grad():
        # Forward pass
        y_pred = model(X_tensor)

        # If model outputs extra dims, flatten to match y_eval
        # e.g. y_pred = y_pred.view_as(y_tensor)

        # Compute MSE
        mse = criterion(y_pred, y_tensor).item()
    return mse

# L₂ norm tells you how “big” your solution is (capacity control).
def l2_norm(w: np.ndarray) -> float:
    return float(np.linalg.norm(w, 2))

def l1_norm(w: np.ndarray) -> float:
    return float(np.linalg.norm(w.reshape(-1), 1))
#L₁/L₂ ratio tells you how many “effective” nonzeros you have (sparsity).
def sparsity_ratio(w: np.ndarray) -> float:
    """
    L1/L2 ratio: higher → more diffuse weights, lower → more concentrated.
    """
    l1 = l1_norm(w)
    l2 = l2_norm(w)
    return l1 / (l2 + 1e-12)

#Kurtosis tells you whether that magnitude is due to a few standout weights or a more uniform spread.
def weight_kurtosis(w):
    # fisher=False → normal distribution has kurtosis = 3
    return kurtosis(w, fisher=False)


def parse_args():
    """
    Parse command line arguments.
    User will have to choose the amount of overparametrization between 110%, 150% and 200%.
    :param overparam: Percentage of features vs samples.
    :return: Parsed arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--overparam",
        choices=[110, 150, 200],
        type=int,
        required=True,
        help="percent of features vs samples",
    )
    return p.parse_args()