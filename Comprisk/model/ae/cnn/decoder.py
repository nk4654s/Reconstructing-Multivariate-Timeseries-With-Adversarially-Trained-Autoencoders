"""Simple Decoder model."""

from typing import List
import torch
from torch import nn


class SequenceDecoder(nn.Module):
    """
    A basic decoder network using a Convolutional network. This class decodes a
    latent space back to the sequence of the encoder.

    Args:
        latent_size (int): The size of the latent vector. Defaults to None.
        kernel_size (int): The size of the convolutional kernels. Defaults to 3.
        num_filters (int): The number of convolutional filters. Defaults to 32.
        dropout_prob (float): The probability of dropout in the model. Defaults to 0.5.
    """

    def __init__(
        self,
        num_features: int,
        num_samples: int,
        kernel_size: int = 3,
        stride: int = 1,
        num_filters: int = 32,
        dropout_prob: float = 0.2,
    ):
        super(SequenceDecoder, self).__init__()

        self._num_samples = num_samples
        self._kernel_size = kernel_size
        self._num_filters = num_filters
        self._stride = stride
        self._dropout_prob = dropout_prob

        # batch, latent size
        self._linear_1 = nn.Linear(
            in_features=self._num_samples,
            out_features=self._num_samples * 2,
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


class CompleteDecoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_samples: int,
        kernel_size: int = 3,
        stride: int = 1,
        num_filters: int = 32,
        dropout_prob: float = 0.2,
        output_padding: List[int] = None
    ) -> None:
        super(CompleteDecoder, self).__init__()

        self._num_features = num_features
        self._num_samples = num_samples
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
            output_padding=output_padding[0]
        )

        self._dropout_1 = nn.Dropout(self._dropout_prob)

        self._conv_trans_2 =  nn.ConvTranspose2d(
            in_channels=self._num_filters,
            out_channels=1,
            kernel_size=(self._kernel_size, self._num_features),
            stride=(self._stride, 1),
            padding=0,
            output_padding=(output_padding[1], 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # change x shape to: batch, 1, latent space
        x = torch.unsqueeze(x, dim=1)
        x = self._conv_trans_1(x)
        x = self._dropout_1(x)
        x = torch.relu(x)

        # change x shape to: batch, channels, sequence, features
        x = torch.unsqueeze(x, dim=-1)
        x = self._conv_trans_2(x)

        # change x shape to: batch, sequence, features
        x = torch.squeeze(x, dim=1)
        return x
