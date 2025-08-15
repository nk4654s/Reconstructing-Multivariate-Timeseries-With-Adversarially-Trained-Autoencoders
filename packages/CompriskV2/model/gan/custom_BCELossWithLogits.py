import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CustomBCELossWithLogits(nn.BCEWithLogitsLoss):
    def __init__(self, weight: float = 1.0):
        super(CustomBCELossWithLogits, self).__init__()
        self._weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = torch.sigmoid(input)
        eps = 1e-7
        input = torch.clamp(input, eps, 1 - eps) # prevent sigmoid ouput of exactly 1 or 0
        loss = - self._weight * (target * torch.log(input) - (1-target) * torch.log(1 - input))
        return  loss.mean()