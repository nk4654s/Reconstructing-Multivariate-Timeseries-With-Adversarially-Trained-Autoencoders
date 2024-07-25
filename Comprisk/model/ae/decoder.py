"""Simple Decoder model."""

import torch
from torch import nn


class Decoder(nn.Module):
    """
    A basic decoder network using a Fully Connected Neural Network (FCNN).

    Args:
        num_features (int): The number of features in the input data. Defaults to 10.
        latent_size (int): The size of the latent vector. Defaults to None.
        output_size (int): The size of the output data. Defaults to None.

    Attributes:
        latent_size (int): The size of the latent vector.
        output_size (int): The size of the output data.
    """

    def __init__(
        self,
        latent_size: int = None,
        kernel_size: int = 3,
        num_filters: int = 32,
        dropout_prob: float = 0.2,
    ):
        super(Decoder, self).__init__()

        self._latent_size = latent_size
        self._kernel_size = kernel_size
        self._num_filters = num_filters
        self._dropout_prob = dropout_prob

        # batch, latent size
        self._linear_1 = nn.Linear(
            in_features=self._latent_size,
            out_features=self._latent_size * 2,
        )
        self._dropout_1 = nn.Dropout(p=self._dropout_prob)
        self._conv_transpose_1 = nn.ConvTranspose1d(
            in_channels=self._num_filters,
            out_channels=1,
            kernel_size=(self._kernel_size, self.latent_size),
            stride=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the decoder network.

        Args:
            x (torch.Tensor): The input tensors.

        Returns:
            torch.Tensor: The decoded output.
        """
        # batch, convolutional features
        x = self._linear_1(x)
        x = self._dropout_1(x)
        x = torch.relu(x)

        # batch, sequence, features
        x = self._conv_transpose_1(x)
        return x
