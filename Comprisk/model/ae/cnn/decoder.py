"""Simple Decoder model."""

import torch
from torch import nn


class Decoder(nn.Module):
    """
    A basic decoder network using a Convolutional network.

    Args:
        latent_size (int): The size of the latent vector. Defaults to None.
        kernel_size (int): The size of the convolutional kernels. Defaults to 3.
        num_filters (int): The number of convolutional filters. Defaults to 32.
        dropout_prob (float): The probability of dropout in the model. Defaults to 0.5.
    """

    def __init__(
        self,
        num_features: int,
        latent_size: int,
        kernel_size: int = 3,
        stride: int = 1,
        num_filters: int = 32,
        dropout_prob: float = 0.2,
    ):
        super(Decoder, self).__init__()

        self._latent_size = latent_size
        self._kernel_size = kernel_size
        self._num_filters = num_filters
        self._stride = stride
        self._dropout_prob = dropout_prob

        # batch, latent size
        self._linear_1 = nn.Linear(
            in_features=self._latent_size,
            out_features=self._latent_size * 2,
        )
        self._dropout_1 = nn.Dropout(p=self._dropout_prob)
        self._conv_transpose_1 = nn.ConvTranspose1d(
            in_channels=self._num_filters,
            out_channels=num_features,
            kernel_size=self._kernel_size,
            stride=self._stride
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

        # swap sequence and features
        x = torch.swapaxes(x, 1, 2)
        x = self._conv_transpose_1(x)
        x = torch.swapaxes(x, 1, 2)
        return x
