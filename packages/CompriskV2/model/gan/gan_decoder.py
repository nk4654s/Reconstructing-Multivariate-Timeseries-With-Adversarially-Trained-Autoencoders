from typing import Tuple

import torch

from CompriskV2.model.auto_encoder import DecoderBase


class GANDecoder(DecoderBase):
    """The decoder part which is based on the default decoder base class. The decoder uses a single LSTM
    module and a simple linear layer by pytorch, hence the configuration possible are:
    Args:
        num_hidden_layers (int): The number of LSTM layers.
        num_hidden_dim (int): The number of nodes within one LSTM layer.
        dropout_prob (float): The probability a neuron is disabled.
        num_features_in (int): The number of features which the model receives. This should equal the encoder part.
        num_features_out (int): The number of features which the model outputs.
    """
    def __init__(
        self,
        num_features_out: int,
        ):
        super().__init__()

        self._num_features_out = num_features_out
        self._activation = torch.nn.LeakyReLU()
        'aten::max_unpool1d ist not supported by onnx so this is a workaround that should accomplish the same'

        """# 1 * 352
        self._unpool_layer1 = torch.nn.ConvTranspose1d(
            in_channels=self._num_features_out * 16,
            out_channels=self._num_features_out * 16,
            kernel_size=17,
            padding=1,
        ) # 15 * 352

        self._conv_layer1 = torch.nn.ConvTranspose1d(
            in_channels=self._num_features_out * 16,
            out_channels=self._num_features_out * 8 ,
            kernel_size=3,
            padding=1,
        ) # 15 * 176
"""
        self._unpool_layer2 = torch.nn.ConvTranspose1d(
            in_channels=self._num_features_out * 8,
            out_channels=self._num_features_out * 8,
            kernel_size=18,
            padding=1,
        )  # 30 * 176

        self._conv_layer2 = torch.nn.ConvTranspose1d(
            in_channels=self._num_features_out * 8,
            out_channels=self._num_features_out * 4,
            kernel_size=3,
            padding=1,
        )  # 30 * 88

        self._unpool_layer3 = torch.nn.ConvTranspose1d(
            in_channels=self._num_features_out * 4,
            out_channels=self._num_features_out * 4,
            kernel_size=18,
            padding=1,
        )  # 45 * 88

        self._conv_layer3 = torch.nn.ConvTranspose1d(
            in_channels=self._num_features_out * 4,
            out_channels=self._num_features_out * 2,
            kernel_size=3,
            padding=1,
        )  # 45 * 44

        self._unpool_layer4 = torch.nn.ConvTranspose1d(
            in_channels=self._num_features_out * 2,
            out_channels=self._num_features_out * 2,
            kernel_size=18,
            padding=1,
        )  # 60 * 44

        self._conv_layer4 = torch.nn.ConvTranspose1d(
            in_channels=self._num_features_out * 2,
            out_channels=self._num_features_out,
            kernel_size=3,
            padding=1,
        )  # 60 * 22

        self._output_layer = torch.nn.Linear(  # linear activation layer
            in_features=self._num_features_out,
            out_features=self._num_features_out,
            bias=False
        )



    def forward(self, batch: torch.Tensor) -> torch.Tensor:

        #output = self._unpool_layer1(batch)
        #output = self._conv_layer1(output)
        #output = self._activation(output)

        output = self._unpool_layer2(batch)
        output = self._conv_layer2(output)
        output = self._activation(output)

        output = self._unpool_layer3(output)
        output = self._conv_layer3(output)
        output = self._activation(output)

        output = self._unpool_layer4(output)
        output = self._conv_layer4(output)
        output = self._activation(output)

        output = torch.swapaxes(output, -2, -1)
        output = self._output_layer(output)

        return output


if __name__ == "__main__":
    numbers = torch.rand(1,88,15)
    model = GANDecoder(0, 22)
    y = model(numbers)
    assert y.shape == (1,60,22)
