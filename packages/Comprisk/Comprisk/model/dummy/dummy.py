"""This dummy model does nothing except taking the input batch and multiplying a factor to it. This class
is meant to test the Comprisk package."""

from typing import List

import torch

from Comprisk.model import BaseModel


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
    ):
        super().__init__(features_in=features_in, features_out=features_out)
        self._factor = factor

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch * self._factor
