"""This module implements a base AutoEncoder class."""
from typing import Dict
import torch
from torch import nn

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
        decoder_kwargs: nn.Module,
        weather_bypass: bool = False,
        kwargs: Dict = None,
    ):
        kwargs = kwargs if kwargs is not None else {}
        super().__init__(**kwargs)

        self._bypass = weather_bypass
        self._encoder = Encoder(**encoder_kwargs)
        self._decoder = Decoder(**decoder_kwargs)

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
        return decoded

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
