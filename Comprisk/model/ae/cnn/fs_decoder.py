"""Simple Decoder model."""

from typing import Dict, List
import torch
from torch import nn

from Comprisk.model.ae import DecoderBase


class FSDecoder(DecoderBase):
    """ The feature sequence decoder is the matching counterpart to the
    `FSEncoder` class. It takes in the encoded feature sequence and
    reconstructs it back to the original size.

    Args:
        num_features (int): The number of features in the encoded sequence.
        kernel_size (int): The size of the convolutional kernels.
        stride (int): The stride of the convolutional kernels.
        num_filters (int): The number of filters to use for each layer.
        dropout_prob (float): Probability of zeroing out activations.
    """
    def __init__(
        self,
        num_features: int,
        kernel_size: int = 3,
        stride: int = 1,
        num_filters: int = 32,
        dropout_prob: float = 0.2,
    ) -> None:
        super(FSDecoder, self).__init__()

        self._num_features = num_features
        self._kernel_size = kernel_size
        self._num_filters = num_filters
        self._stride = stride
        self._dropout_prob = dropout_prob

        self._conv_trans_1 = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=self._num_filters,
            kernel_size=self._kernel_size // stride,
            stride=1,
            padding=0,
        )

        self._dropout_1 = nn.Dropout(self._dropout_prob)

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=self._num_filters,
            out_channels=1,
            kernel_size=(self._kernel_size, self._num_features),
            stride=(self._stride, 1),
            padding=0,
        )

    @property
    def output_keys(self) -> List[str]:
        return ["sensors"]

    def forward(self, z: torch.Tensor, inputs: Dict[str, torch.tensor]) -> torch.Tensor:
        output_size = inputs["sensors"].size()[1:]

        # change x shape to: batch, 1, latent space
        x = torch.unsqueeze(z, dim=1)
        x = self._conv_trans_1(x)
        x = self._dropout_1(x)
        x = torch.relu(x)

        # change x shape to: batch, channels, sequence, features
        x = torch.unsqueeze(x, dim=-1)
        x = self._conv_trans_2(x, output_size=output_size)

        # change x shape to: batch, sequence, features
        x = torch.squeeze(x, dim=1)
        return x
