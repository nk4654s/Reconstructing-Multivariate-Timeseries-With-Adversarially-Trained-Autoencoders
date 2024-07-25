"""This module implements a base AutoEncoder class."""
from typing import Any, Dict
import torch

from Comprisk.model import BaseModel
from Comprisk.model.ae.encoder import Encoder
from Comprisk.model.ae.decoder import Decoder


class AutoEncoder(BaseModel):
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
        encoder_kwargs: Dict,
        decoder_kwargs: Dict,
        num_filters: int = None,
        kernel_size: int = None,
        dropout_prob: float = None,
        weather_bypass: bool = False,
        kwargs: Dict = None,
    ):
        kwargs = kwargs if kwargs is not None else {}
        super().__init__(**kwargs)

        self._encoder_kwargs = encoder_kwargs
        self._decoder_kwargs = decoder_kwargs

        # overwrite args
        self._overwrite_args("num_filters", num_filters)
        self._overwrite_args("kernel_size", kernel_size)
        self._overwrite_args("dropout", dropout_prob)

        # build encoder/decoder
        self._bypass = weather_bypass
        self._encoder = Encoder(**encoder_kwargs)

        decoder_kwargs["latent_size"] = self._encoder.latent_size
        self._decoder = Decoder(**decoder_kwargs)

    def _overwrite_args(self, key: str, value: Any) -> None:
        if value is not None:
            self._encoder_kwargs[key] = value
            self._decoder_kwargs[key] = value

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Passes the input through both the encoder and decoder.

        Args:
            inputs (Dict[str, torch.Tensor]): The input tensor.

        Returns:
            torch.Tensor: The output of the autoencoder.
        """
        x = inputs["sensors"]
        if not self._bypass:
            weather = inputs["weather"]
            x = torch.cat([x, weather], dim=1)

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
