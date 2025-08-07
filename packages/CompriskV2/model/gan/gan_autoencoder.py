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
            num_features: int,
            num_features_out: int,
    ) -> None:

        super().__init__()

        self._num_features = num_features
        self._num_features_out = num_features_out

        self._encoder = GANEncoder(
            num_features=self._num_features,
        )
        self._decoder = GANDecoder(
            num_features_out=self._num_features_out
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoder_output = self._encoder(batch)
        #if self.training:
        #    encoder_output = encoder_output + (0.1**0.5)*torch.randn(encoder_output.size()).to(batch.device)
        decoder_output = self._decoder(encoder_output)
        return decoder_output


"""
if __name__ == "__main__":
    numbers = torch.rand(1,60,22)
    model = GANAutoEncoder(22, 0, 22 )
    y = model(numbers)
    assert y.shape == (1,60,22)
"""
