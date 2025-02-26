"""This dummy model does nothing except taking the input batch and multiplying a factor to it. This class
is meant to test the Comprisk package."""

from typing import List

import numpy as np
import torch
from torch import Tensor, nn

from Comprisk.model import BaseModel, NormalizeModule


class DummyModel(BaseModel):
    """Dummy model required to test the Comprisk package.

    Args:
        factor (float): The factor multiplied to the batch input.
    """

    def __init__(
        self,
        factor: float,
        features_in: List[str] = None,
        features_out: List[str] = None,
        enable_normalizer: bool = False
    ):
        super().__init__(features_in=features_in, features_out=features_out)
        data = {
            n: {
                "min": 0,
                "max": 1
            } for n in features_in
        }
        self._normalizer = NormalizeModule(normalize_data=data, features_in=features_in, features_out=features_out)
        if not enable_normalizer:
            self._normalizer.apply = lambda _: _
            self._normalizer.reverse = lambda x, y: (x, y)

        self._factors = nn.Parameter(Tensor(np.array([factor])))

    def forward(self, x: Tensor) -> torch.Tensor:
        return x * self._factors
