import time, random
import numpy as np
import torch
import torch.nn as nn
from ..models import LinearNetModel

def exp_delay(num_workers: int, scale: float = 1e-4) -> None:
    mean_stale = num_workers - 1
    time.sleep(np.random.exponential(mean_stale * scale))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

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
