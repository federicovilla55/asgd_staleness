# model.py
import torch.nn as nn
import torch

class LinearNetModel(nn.Module):
    """
    Simple Linear Neural Network Model

    :param input_dim: Number of model input features.
    :type input_dim: int
    :param bias: Whether to include a bias term in the linear layer.
    :type input_dim: int
    """
    def __init__(self, input_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. 
        It applies a sigmoid activation function to the output of the linear layer.

        :param x: Input tensor of shape `batch_size`, `input_dim`.
        :type x: torch.Tensor
        :return: Output tensor after sigmoid activation.
        :rtype: torch.Tensor
        :raises ValueError: If the input tensor does not have the expected shape.
        """
        return torch.sigmoid(self.linear(x)).squeeze(-1)