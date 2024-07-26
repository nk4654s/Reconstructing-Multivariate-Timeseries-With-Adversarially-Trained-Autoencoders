"""This module implements a base AutoEncoder class."""

from typing import Any, Dict
import torch

from Dataflux.helper import get_logger
from torch.nn.modules import Module

from Comprisk.model.ae.base import AutoEncoderBase
from Comprisk.model.ae.cnn.encoder import Encoder
from Comprisk.model.ae.cnn.decoder import Decoder

logger = get_logger(__name__)


class AutoEncoder(AutoEncoderBase):
    """
    A simple autoencoder class.

    Args:
        encoder_kwargs (Dict): The encoder hyper-parameters.
        decoder_kwargs (Dict): The decoder hyper-parameters.
        weather_bypass (bool): Whether to bypass the weather data directly to the decoder.
        kwargs (Dict): Keyword arguments to pass to the base model class.
    """

    def __init__(
        self,
        encoder_kwargs: Dict = None,
        decoder_kwargs: Dict = None,
        num_filters: int = None,
        kernel_size: int = None,
        stride: int = None,
        dropout_prob: float = None,
        base: Dict = None,
    ):
        base = {} if base is None else base

        # overwrite args
        self._overwrite_args("num_filters", num_filters)
        self._overwrite_args("kernel_size", kernel_size)
        self._overwrite_args("stride", stride)
        self._overwrite_args("dropout_prob", dropout_prob)

        super().__init__(**base)

        self._encoder_kwargs = {} if encoder_kwargs is None else encoder_kwargs
        self._decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs

    def init_encoder(self) -> Module:
        encoder_input = self._num_features["sensors"]
        if not self._bypass:
            encoder_input += self._num_features["weather"]

        return Encoder(num_features=encoder_input, **self._encoder_kwargs)

    def init_decoder(self) -> Module:
        logger.debug("Latent space size is: %s", self._encoder.latent_size)
        return Decoder(
            num_features=self._num_features["targets"],
            latent_size=self._encoder.latent_size,
            **self._decoder_kwargs
        )

    def _overwrite_args(self, key: str, value: Any) -> None:
        if value is not None:
            self._encoder_kwargs[key] = value
            self._decoder_kwargs[key] = value

    @property
    def target(self) -> str:
        """The target name of inputs to predict."""
        return "sensors"

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Passes the input through both the encoder and decoder.

        Args:
            inputs (Dict[str, torch.Tensor]): The input tensor.

        Returns:
            torch.Tensor: The output of the autoencoder.
        """
        # concat inputs when bypass is inactive
        x = inputs["sensors"]
        if not self._bypass:
            weather = inputs["weather"]
            x = torch.cat([x, weather], dim=-1)

        # pass inputs through encoder/decoder
        encoded = self._encoder(x)
        if self._bypass:
            encoded = torch.cat([encoded, inputs["weather"]], dim=-1)

        decoded = self._decoder(encoded)
        return self._activation(decoded)

    def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encodes the input using the encoder network.

        Args:
            inputs (Dict[str, torch.Tensor]): The input tensor.

        Returns:
            torch.Tensor: The encoded output.
        """
        x = inputs["sensors"]
        if not self._bypass:
            weather = inputs["weather"]
            x = torch.cat([x, weather], dim=1)

        return self._encoder(x)

    def decode(self, z: torch.Tensor, x_weather: torch.Tensor = None) -> torch.Tensor:
        """
        Decodes the input using the decoder network.

        Args:
            z (torch.Tensor): The input tensor to be decoded.
            x_weather (torch.Tensor): The weather input tensor when bypassing the encoder.

        Returns:
            torch.Tensor: The decoded output.
        """
        if self._bypass:
            z = torch.cat([z, x_weather], dim=-1)

        return self._decoder(z)
