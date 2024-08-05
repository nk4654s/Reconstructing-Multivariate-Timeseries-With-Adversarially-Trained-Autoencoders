"""This module implements a base AutoEncoder class."""

from torchinfo import summary as torchsummary
from typing import Dict
import numpy as np
import torch

from Dataflux.helper import get_logger

from Comprisk.model.ae.base import AutoEncoderBase
from Comprisk.model.ae.cnn.encoder import SequenceEncoder, CompleteEncoder
from Comprisk.model.ae.cnn.decoder import SequenceDecoder, CompleteDecoder

logger = get_logger(__name__, path="./logs/logfile")


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
        mode: str = "complete",
        num_weather: int = None,
        num_sensors: int = None,
        num_samples: int = None,
        kernel_size: int = None,
        num_filters: int = None,
        stride: int = None,
        dropout_prob: float = None,
        weather_bypass: bool = False,
        base: Dict = None,
    ):
        base = {} if base is None else base
        super().__init__(weather_bypass=weather_bypass, base=base)
        self.save_hyperparameters()

        self._num_weather = num_weather
        self._num_sensors = num_sensors
        self._num_samples = num_samples

        logger.debug("Using '%s' auto-encoder", mode)
        self._encoder_class = SequenceEncoder if mode == "sequence" else CompleteEncoder
        self._decoder_class = SequenceDecoder if mode == "sequence" else CompleteDecoder

        self._encoder = self._encoder_class(
            **{
                "num_features": num_weather + num_sensors,
                "num_samples": num_samples,
                "kernel_size": kernel_size,
                "stride": stride,
                "num_filters": num_filters,
                "dropout_prob": dropout_prob,
            }
        )

        self._decoder = self._decoder_class(
            **{
                "num_features": num_sensors,
                "num_samples": self._encoder.latent_size,
                "kernel_size": kernel_size,
                "stride": stride,
                "num_filters": num_filters,
                "dropout_prob": dropout_prob,
                "output_padding": self._encoder.output_padding
            }
        )

    def summary(self) -> str:
        input_data = {
            "sensors": torch.tensor(
                np.zeros((1, self._num_samples, self._num_sensors), dtype=np.float32)
            ),
            "weather": torch.tensor(
                np.zeros((1, self._num_samples, self._num_weather), dtype=np.float32)
            ),
        }
        return str(torchsummary(self, input_data={"inputs": input_data}))

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
