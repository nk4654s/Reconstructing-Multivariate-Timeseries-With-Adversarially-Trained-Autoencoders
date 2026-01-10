"""This module implements a basic GAN (Generative Adversarial Network) class and the base classes for an
encoder, decoder, autoencoder and classifier."""

from typing import Tuple

import torch

from torch import Tensor, nn

from CompriskV2.model.gan.gan_decoder import GANDecoder
from CompriskV2.model.gan.gan_encoder import GANEncoder


class GANAutoEncoder(torch.nn.Module):

    def __init__(
            self,
            num_features_in: int,
            num_features_out: int,
            weight_init,
            dropout_rate: float,
            output_channels_first_stage: int = 22,
            output_channels_second_stage: int = 22,
    ) -> None:

        super().__init__()

        self._num_features_in = num_features_in
        self._num_features_out = num_features_out
        self._weight_init = weight_init
        self._dropout_rate = dropout_rate
        self._output_channels_first_stage = output_channels_first_stage
        self._output_channels_second_stage = output_channels_second_stage

        self._encoder = GANEncoder(
            num_features_in=self._num_features_in,
            weight_init = self._weight_init,
            output_channels_first_stage=self._output_channels_first_stage,
            output_channels_second_stage=self._output_channels_second_stage,
        )
        self._decoder = GANDecoder(
            num_features_out=self._num_features_out,
            weight_init = self._weight_init,
            output_channels_first_stage=self._output_channels_first_stage,
            output_channels_second_stage=self._output_channels_second_stage,
        )

        self.dropout_layer = nn.Dropout(self._dropout_rate)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoder_output = self._encoder(batch)
        if self.training:
            encoder_output = self.dropout_layer(encoder_output)
        decoder_output = self._decoder(encoder_output)
        return decoder_output


"""
if __name__ == "__main__":
    numbers = torch.rand(1,60,22)
    model = GANAutoEncoder(22, 0, 22 )
    y = model(numbers)
    assert y.shape == (1,60,22)
"""
