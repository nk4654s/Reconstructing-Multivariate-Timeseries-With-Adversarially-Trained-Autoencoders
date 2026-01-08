from typing import Tuple

import torch

from Comprisk.model.auto_encoder import DecoderBase


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
        weight_init,
        output_channels_first_stage: int = 22,
        output_channels_second_stage: int = 22,
        conv_kernel_size: int = 3,
        ):
        super().__init__()

        self._num_features_out = num_features_out
        self._activation = torch.nn.LeakyReLU()
        self._weight_init = weight_init
        self._output_channels_first_stage = output_channels_first_stage
        self._output_channels_second_stage = output_channels_second_stage
        self._conv_kernel_size = conv_kernel_size
        'aten::max_unpool1d is not supported by onnx so this is a workaround that should accomplish the same'

        # (22, 15)

        self._unpool_layer1 = torch.nn.ConvTranspose1d(
            in_channels=self._output_channels_second_stage,
            out_channels=self._output_channels_first_stage,
            kernel_size=self._conv_kernel_size,
            stride=2,
            padding=1,
            output_padding=1,
        )  # (22, 30)


        self._conv_layer1 = torch.nn.ConvTranspose1d(
            in_channels=self._output_channels_first_stage,
            out_channels=self._output_channels_first_stage,
            kernel_size=self._conv_kernel_size,
            padding=1,
        )  # (22, 30)

        self._unpool_layer2 = torch.nn.ConvTranspose1d(
            in_channels=self._output_channels_first_stage,
            out_channels=self._num_features_out,
            kernel_size=self._conv_kernel_size,
            stride=2,
            padding=1,
            output_padding=1,
        )  # (22, 60)

        self._conv_layer2 = torch.nn.ConvTranspose1d(
            in_channels=self._num_features_out,
            out_channels=self._num_features_out,
            kernel_size=self._conv_kernel_size,
            padding=1,
        )  # (22, 60)

        self._output_layer = torch.nn.Linear(  # linear activation layer
            in_features=self._num_features_out,
            out_features=self._num_features_out,
        )

        #########################
        # Weight Initialization #
        #########################
        self._weight_init(self._conv_layer1.weight)
        self._weight_init(self._conv_layer2.weight)
        self._weight_init(self._output_layer.weight)


    def forward(self, batch: torch.Tensor) -> torch.Tensor:

        output = self._unpool_layer1(batch)
        output = self._conv_layer1(output)
        output = self._activation(output)

        output = self._unpool_layer2(output)
        output = self._conv_layer2(output)
        output = self._activation(output)

        output = torch.swapaxes(output, -2, -1)
        output = self._output_layer(output)

        return output
