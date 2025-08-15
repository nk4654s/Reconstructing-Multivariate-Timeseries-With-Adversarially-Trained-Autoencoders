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
        input = torch.Tensor(nn.Sigmoid()(input))
        loss = - self._weight * (target * torch.log(input) - (1-target) * torch.log(1 - input))
        if math.isnan(loss.mean()):
            nan_idx = next((i for i, v in enumerate(loss) if v != v), -1)
            test = loss[nan_idx]
            print(nan_idx)
        return  loss.mean()