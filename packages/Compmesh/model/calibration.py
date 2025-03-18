import numpy as np
from torch import nn, Tensor


class LinearCalibration(nn.Module):
    """The LinearCalibration model takes an input and applies a linear function to it. The slope and intercept of
    the function are learnable and adjusted according to the calculated loss and backpropagation.

    Args:
        output_features (int): The number of features to output which is required to initialize the linear function.
    """
    def __init__(self, output_features: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slope = nn.Parameter(Tensor(np.ones(output_features)))
        self._intercept = nn.Parameter(Tensor(np.zeros(output_features)))

    def forward(self, x: Tensor) -> Tensor:
        y_final = self._slope * x + self._intercept
        return y_final