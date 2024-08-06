"""This module implements a base AutoEncoder class."""

from typing import Dict, List
from torchinfo import summary as torchsummary
from torch import nn
import numpy as np
import torch

from Dataflux.helper import get_logger

from Comprisk.model.ae.base import AutoEncoderBase
from Comprisk.model.ae.cnn.encoder import Encoder
from Comprisk.model.ae.cnn.decoder import Decoder

logger = get_logger(__name__, path="./logs/logfile")


class AutoEncoder(AutoEncoderBase):
    """
    A simple autoencoder class.

    Args:
        encoder_kwargs (Dict): The encoder hyper-parameters.
        decoder_kwargs (Dict): The decoder hyper-parameters.
        weather_bypass (int): Hidden size of the weather bypass reconstruction. Defaults
            to 0 which deactivates the bypass.
        kwargs (Dict): Keyword arguments to pass to the base model class.
    """

    def __init__(
        self,
        num_weather: int = None,
        num_sensors: int = None,
        num_samples: int = None,
        kernel_size: int = None,
        num_filters: int = None,
        stride: int = None,
        dropout_prob: float = None,
        bypass_hidden_size: int = 0,
        base: Dict = None,
    ):
        base = {} if base is None else base
        super().__init__(weather_bypass=bypass_hidden_size>0, base=base)
        self.save_hyperparameters()

        self._num_weather = num_weather
        self._num_sensors = num_sensors
        self._num_samples = num_samples

        num_encoder_in = num_sensors if self._bypass else num_weather + num_sensors
        self._encoder = Encoder(
            **{
                "num_features": num_encoder_in,
                "num_samples": num_samples,
                "kernel_size": kernel_size,
                "stride": stride,
                "num_filters": num_filters,
                "dropout_prob": dropout_prob,
            }
        )

        self._decoder = Decoder(
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

        if self._bypass:
            self._linear_1 = nn.Linear(
                num_sensors + num_weather,
                bypass_hidden_size,
            )
            self._dropout_1 = nn.Dropout(dropout_prob)
            self._linear_2 = nn.Linear(
                bypass_hidden_size,
                num_sensors + num_weather
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
    def target(self) -> str | List[str]:
        """The target name of inputs to predict."""
        return ["sensors", "weather"] if self._bypass else "sensors"

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Passes the input through both the encoder and decoder.

        Args:
            inputs (Dict[str, torch.Tensor]): The input tensor.

        Returns:
            torch.Tensor: The output of the autoencoder.
        """
        x = self.encode(inputs)
        x = self.decode(x, inputs["weather"])

        return self._activation(x)

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
            x = torch.cat([x, weather], dim=-1)

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
        x = self._decoder(z)
        if self._bypass:
            x = torch.cat([x, x_weather], dim=-1)

            x = torch.relu(x)
            x = self._linear_1(x)
            x = self._dropout_1(x)
            x = torch.relu(x)
            x = self._linear_2(x)

        return x
