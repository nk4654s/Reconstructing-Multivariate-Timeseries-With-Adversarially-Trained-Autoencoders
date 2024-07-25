"""Simple Encoder model."""

import torch
from torch import nn


class Encoder(nn.Module):
    """
    A basic encoder network using a Convolutional Neural Network (CNN).

    Args:
        num_features (int): The number of features in the input data. Defaults to 10.
        kernel_size (int): The size of the convolutional kernels. Defaults to 3.
        num_filters (int): The number of filters used in the convolutional layers. Defaults to 64.
        dropout_prob (float): The probability of dropout used in the network. Defaults to 0.2.
    """

    def __init__(
        self,
        num_features: int = 10,
        kernel_size: int = 3,
        num_filters: int = 32,
        dropout_prob: float = 0.2,
    ):
        super(Encoder, self).__init__()

        self._num_features = num_features
        self._kernel_size = kernel_size
        self._num_filters = num_filters
        self._dropout_prob = dropout_prob

        self._conv_1 = nn.Conv1d(
            self._num_features,
            self._num_filters,
            kernel_size=self._kernel_size,
            stride=1,
        )
        self._size = self._num_filters * (self._num_features - self._kernel_size + 1)
        self._dropout_1 = nn.Dropout(p=self._dropout_prob)
        self._linear_1 = nn.Linear(
            self._num_filters,
            self._num_filters // 2
        )

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
        #x = torch.flatten(x, start_dim=1)
        x = torch.swapaxes(x, 1, 2)

        # batch, convolutional features
        x = self._linear_1(x)
        return x
