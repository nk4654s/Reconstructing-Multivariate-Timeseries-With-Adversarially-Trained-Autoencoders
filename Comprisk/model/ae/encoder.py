"""Simple Encoder model."""
import torch
from torch import nn, F


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
        num_samples: int = 10000,
        kernel_size: int = 3,
        num_filters: int = 64,
        dropout_prob: float = 0.2,
    ):
        super(Encoder, self).__init__()

        self._num_features = num_features
        self._num_samples = num_samples
        self._kernel_size = kernel_size
        self._num_filters = num_filters
        self._dropout_prob = dropout_prob

        # window shape (H, W, D): (num_samples, num_features, 1)
        self._conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=self._num_filters,
            kernel_size=(self._kernel_size, self._num_samples),
            stride=None,
            padding="same",
        )

        self._dropout_1 = nn.Dropout(p=self._dropout_prob)
        self._linear_1 = nn.Linear(
            self._num_filters * (self._num_features - self._kernel_size + 1),
            128
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the encoder network.

        Args:
            x (torch.Tensor): The input tensors.

        Returns:
            torch.Tensor: The encoded output.
        """
        x = self._conv_1(x)
        x = F.relu(x)
        x = torch.max_pool1d(x, kernel_size=2)
        x = self._dropout_1(x)
        x = x.view(-1, 128)
        x = self._linear_1(x)
        return x
