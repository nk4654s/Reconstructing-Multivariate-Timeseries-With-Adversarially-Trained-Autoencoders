"""Simple Encoder model."""

from typing import List
import torch
from torch import nn


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
            kernel_size=kernel_size // stride,
            stride=1,
            padding=0
        )

        # calculate encoder output sizes
        size_1 = int(((num_samples - kernel_size) / stride) + 1)
        self._latent = int(((size_1 - (kernel_size // stride)) / 1) + 1)

        # calculate decoder output sizes to find output_padding size
        revert_1 = (self._latent - 1) * 1 + (kernel_size // stride)
        out_padding_1 = abs(size_1 - revert_1)

        revert_2 = (revert_1 + out_padding_1 - 1) * stride + kernel_size
        out_padding_2 = abs(num_samples - revert_2)
        self._output_padding = [out_padding_1, out_padding_2]

    @property
    def latent_size(self) -> int:
        """Returns the size of the latent vector."""
        return self._latent

    @property
    def output_padding(self) -> List[int]:
        """Returns the output padding size for each encoder layer, which is
        required in the decoder to restore to the correct shape."""
        return self._output_padding

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
