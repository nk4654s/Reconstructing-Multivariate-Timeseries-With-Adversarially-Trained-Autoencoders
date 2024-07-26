"""This module implements a base AutoEncoder class."""

from typing import Dict
import torch

from Dataflux.helper import get_logger

from Comprisk.model import BaseModel

logger = get_logger(__name__)


class AutoEncoderBase(BaseModel):
    """
    The auto encoder base class implements the forward and predict methods, as well as initializer
    function to define the encoder/decoder.

    Args:
        weather_bypass (bool): Whether to bypass the weather data directly to the decoder.
        kwargs (Dict): Keyword arguments to pass to the base model class.
    """

    def __init__(
        self,
        weather_bypass: bool = False,
        base: Dict = None,
    ):
        base = {} if base is None else base
        super().__init__(**base)

        # build encoder/decoder
        self._bypass = weather_bypass
        self._encoder = self.init_encoder()
        self._decoder = self.init_decoder()

    def init_encoder(self) -> torch.nn.Module:
        """Override this method in a child class to build the encoder.

        Returns:
            torch.nn.Module: The encoder model.
        """
        raise NotImplementedError

    def init_decoder(self) -> torch.nn.Module:
        """Override this method in a child class to build the decoder.

        Returns:
            torch.nn.Module: The decoder model.
        """
        raise NotImplementedError

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
