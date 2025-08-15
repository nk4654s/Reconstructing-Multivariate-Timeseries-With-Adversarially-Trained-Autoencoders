import torch
from torch.nn.utils import spectral_norm

from CompriskV2.model import BaseModel


def check_tensor(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

class GANClassificator(torch.nn.Module):
    """
    simple num_features to 1 classificator with 3 dense layers for now
    """
    def __init__(
            self,
            num_features: int,
            sequence_input_size: int,
            num_layers: int = 1, )-> None:
        super().__init__()
        self._num_features = num_features
        self._sequence_input_size = sequence_input_size
        self._activation = torch.nn.LeakyReLU()

        self._conv_layer1 = spectral_norm(torch.nn.Conv1d(
            in_channels=self._num_features,
            out_channels=self._num_features * 4,
            kernel_size=3, padding=1,
        )) # 60 * 88

        self._pool1 = torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=1) # 60 * 88

        self._conv_layer2 = spectral_norm(torch.nn.Conv1d(
            in_channels=self._num_features * 4,
            out_channels=self._num_features * 8,
            kernel_size=3,
            padding=1,
        )) # 60 * 176

        self._pool2 = torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=1) # 60 * 176

        self._conv_layer3 = spectral_norm(torch.nn.Conv1d(
            in_channels=self._num_features * 8,
            out_channels=self._num_features,
            kernel_size=3, padding=1,
        )) # 60 * 22

        self._output_layer = spectral_norm(torch.nn.Linear(
            in_features=self._num_features * 60, # = 1320
            out_features=1,
        )) # 1

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = torch.swapaxes(batch, -2, -1)

        output = self._conv_layer1(batch)
        output = self._activation(output)
        output = self._pool1(output)

        output = self._conv_layer2(output)
        output = self._activation(output)
        output = self._pool2(output)

        output = self._conv_layer3(output)
        output = self._activation(output)
        output = output.view(batch.size(0), -1, 1) # 1320 * 1

        output = torch.swapaxes(output, -2, -1)

        output = self._output_layer(output)
        #output = torch.nn.Sigmoid()(output)
        return output
