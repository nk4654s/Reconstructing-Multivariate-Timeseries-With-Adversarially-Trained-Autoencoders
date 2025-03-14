import numpy as np
from torch import nn, Tensor


class LinearCalibration(nn.Module):
    def __init__(self, output_features: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slope = nn.Parameter(Tensor(np.ones(output_features)))
        self._intercept = nn.Parameter(Tensor(np.zeros(output_features)))

    def forward(self, x: Tensor) -> Tensor:
        y_final = self._slope * x + self._intercept
        return y_final