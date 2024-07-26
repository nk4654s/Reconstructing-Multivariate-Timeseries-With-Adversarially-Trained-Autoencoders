"""Simple Encoder model."""

import torch
from torch import nn


class SequenceEncoder(nn.Module):
    """
    A basic encoder network using a Convolutional Neural Network (CNN). Encodes a sequence of
    features into a fixed length vector.

    Args:
        num_features (int): The number of features in the input data. Defaults to 10.
        kernel_size (int): The size of the convolutional kernels. Defaults to 3.
        num_filters (int): The number of filters used in the convolutional layers. Defaults to 64.
        dropout_prob (float): The probability of dropout used in the network. Defaults to 0.2.
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
        super(SequenceEncoder, self).__init__()

        self._num_features = num_features
        self._kernel_size = kernel_size
        self._num_filters = num_filters
        self._stride = stride
        self._dropout_prob = dropout_prob

        self._conv_1 = nn.Conv1d(
            self._num_features,
            self._num_filters,
            kernel_size=self._kernel_size,
            stride=self._stride,
        )

        self._size = self._num_filters * (self._num_features - self._kernel_size + 1)
        self._dropout_1 = nn.Dropout(p=self._dropout_prob)
        self._linear_1 = nn.Linear(self._num_filters, self._num_filters // 2)

    @property
    def latent_size(self) -> int:
        """Returns the size of the latent vector."""
        return self._num_filters // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the encoder network.

        Args:
            x (torch.Tensor): The input tensors.

        Returns:
            torch.Tensor: The encoded output.
        """
        # swap sequence and features
        x = torch.swapaxes(x, 1, 2)

        x = self._conv_1(x)
        x = torch.relu(x)

        x = self._dropout_1(x)
        # x = torch.flatten(x, start_dim=1)
        x = torch.swapaxes(x, 1, 2)

        # batch, convolutional features
        x = self._linear_1(x)
        return x


class CompleteEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_samples: int,
        kernel_size: int = 3,
        stride: int = 1,
        num_filters: int = 32,
        dropout_prob: float = 0.2,
    ) -> None:
        super(CompleteEncoder, self).__init__()

        self._num_features = num_features
        self._num_samples = num_samples
        self._kernel_size = kernel_size
        self._num_filters = num_filters
        self._stride = stride
        self._dropout_prob = dropout_prob

        self._cnn_1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(kernel_size, num_features),
            stride=(stride, 1),
            padding=0
        )

        self._dropout_1 = nn.Dropout(self._dropout_prob)

        self._cnn_2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )

        size_1 = ((num_samples - kernel_size) / stride) + 1
        self._latent = ((size_1 - kernel_size) / stride) + 1

    @property
    def latent_size(self) -> int:
        """Returns the size of the latent vector."""
        return self._latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # change x shape to: batch, 1, sequence, features
        x = torch.unsqueeze(x, dim=1)
        x = self._cnn_1(x)
        x = self._dropout_1(x)
        x = torch.relu(x)

        # change x shape to: batch, channels, sequence
        x = torch.squeeze(x, dim=-1)
        x = self._cnn_2(x)

        # change x shape to: batch, sequence
        x = torch.squeeze(x, dim=1)
        return x
