import torch
from torch.nn.utils import spectral_norm

from CompriskV2.model import BaseModel


def check_tensor(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

class GANClassificator(torch.nn.Module):

    def __init__(
            self,
            num_features_in: int,
            sequence_in: int,
            output_linear_hidden_dim: int,
            weight_init,
            output_channels_first_stage: int = 22,
            output_channels_second_stage: int = 22,
            conv_kernel_size: int = 3,
    )-> None:
        super().__init__()
        self._num_features_in = num_features_in
        self._sequence_in = sequence_in
        self._output_linear_hidden_dim = output_linear_hidden_dim
        self._activation = torch.nn.LeakyReLU()
        self._weight_init = weight_init
        self._output_channels_first_stage = output_channels_first_stage
        self._output_channels_second_stage = output_channels_second_stage
        self._conv_kernel_size = conv_kernel_size

        self._conv_layer1 = torch.nn.Conv1d(
            in_channels=self._num_features_in,
            out_channels=self._output_channels_first_stage,
            kernel_size=self._conv_kernel_size,
            stride=1,
            padding=1
        )  # (22, 60)

        self._pool_layer1 = torch.nn.AvgPool1d(
            kernel_size=self._conv_kernel_size,
            stride=2,
            padding=1,
        )  # (22, 30)

        self._conv_layer2 = torch.nn.Conv1d(
            in_channels=self._output_channels_first_stage,
            out_channels=self._output_channels_second_stage,
            kernel_size=self._conv_kernel_size,
            stride=1,
            padding=1
        )  # (22, 30)

        self._pool_layer2 = torch.nn.AvgPool1d(
            kernel_size=self._conv_kernel_size,
            stride=2,
            padding=1,
        )  # (22, 15)

        self._output_linear = torch.nn.Sequential(
            torch.nn.Linear(int(self._sequence_in / 4) * self._output_channels_second_stage, self._output_linear_hidden_dim),
            self._activation,
            torch.nn.Linear(self._output_linear_hidden_dim, self._output_linear_hidden_dim),
            self._activation,
            torch.nn.Linear(self._output_linear_hidden_dim, 1)  # logits
        )

        #########################
        # Weight Initialization #
        #########################
        self._weight_init(self._conv_layer1.weight)
        self._weight_init(self._conv_layer2.weight)
        for layer in self._output_linear:
            if isinstance(layer, torch.nn.Linear):
                self._weight_init(layer.weight)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = torch.swapaxes(batch, -2, -1)

        output = self._conv_layer1(batch)
        output = self._activation(output)
        output = self._pool_layer1(output)

        output = self._conv_layer2(output)
        output = self._activation(output)
        output = self._pool_layer2(output)

        output = torch.swapaxes(output, -2, -1)

        output = output.flatten(start_dim=1, end_dim=-1) # flatten dataspace to

        output = self._output_linear(output)

        return output
