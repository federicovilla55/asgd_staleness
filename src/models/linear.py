import torch
import torch.nn as nn

class LinearNetModel(nn.Module):
    """
    Simple Linear Regression Model

    :param input_dim: Number of model input features.
    :type input_dim: int
    :param bias: Whether to include a bias term in the linear layer.
    :type input_dim: int
    """
    def __init__(self, input_dim: int, bias: bool = True):
        super().__init__()
        # A single linear layer mapping input_dim features to a single output
        self.linear = nn.Linear(input_dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Applies a linear transformation without any activation, returning the raw output.

        :param x: Input tensor of shape (batch_size, input_dim).
        :type x: torch.Tensor
        :return: Output tensor of shape (batch_size,).
        :rtype: torch.Tensor
        """
        # Linear layer returns (batch_size, 1), so squeeze to (batch_size,)
        return self.linear(x).squeeze(-1)