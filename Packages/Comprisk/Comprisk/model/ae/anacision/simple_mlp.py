"""The anacision autoencder uses a simple multi layer perceptron."""
from typing import Dict, List
import torch
from torch import nn

from Comprisk.model.ae.base import EncoderBase, DecoderBase


class AnacisionEncoder(EncoderBase):
    """The anacision encoder uses a single layer to reduce a single time step's dimension.

    Args:
        num_input (int): The number of sensors.
        num_latent (int): The number of latent variables.
    """
    def __init__(self, num_input: int, num_latent: int):
        super().__init__()
        self._linear = nn.Linear(
            num_input,
            num_latent
        )

    def forward(self, batch):
        return self._linear(batch)

    @property
    def input_keys(self) -> List[str]:
        return ["sensors"]


class AnacisionDecoder(DecoderBase):
    """The anacision decoder uses a single layer to expand the latent space back into the original space.

    Args:
        num_latent (int): The number of latent variables.
        num_output (int): The number of sensors.
    """
    def __init__(self, num_latent: int, num_output: int):
        super().__init__()
        self._linear = nn.Linear(
            num_latent,
            num_output
        )

    def forward(self, z: torch.Tensor, inputs: Dict[str, torch.tensor]) -> torch.Tensor:
        return self._linear(z)

    @property
    def output_keys(self) -> List[str]:
        return ["sensors"]
