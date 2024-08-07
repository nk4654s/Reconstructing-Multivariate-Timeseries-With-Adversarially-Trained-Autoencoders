"""This module implements a basic AutoEncoder class and the base classes for an
encoder and decoder."""

from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary as torchsummary

from Dataflux.pytorch.metadata import Metadata
from Dataflux.helper import get_logger
from Comprisk.model import BaseModel

logger = get_logger(__name__, path="./logs/logfile")


class EncoderBase(nn.Module):
    """
    The encoder base class implements the forward and predict methods, as well as initializer
    function to define the encoder.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(EncoderBase, self).__init__()

    def forward(self, batch: torch.tensor) -> torch.Tensor:
        """Implements the forward pass of the encoder."""
        raise NotImplementedError()

    @property
    def input_keys(self) -> List[str]:
        """Returns a list of input keys for the encoder."""
        return ["sensors", "weather"]


class DecoderBase(nn.Module):
    """
    The decoder base class implements the forward and predict methods, as well as initializer
    function to define the decoder.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(DecoderBase, self).__init__()

    def forward(self, z: torch.Tensor, inputs: Dict[str, torch.tensor]) -> torch.Tensor:
        """Implements the forward pass of the decoder."""
        raise NotImplementedError()

    @property
    def output_keys(self) -> List[str]:
        """The output keys to know what the target is."""
        return ["sensors", "weather"]


class AutoEncoder(BaseModel):
    """The AutoEncoder class uses a given encoder and decoder model to learn a latent
    space representation of the input data. Usually, the latent space is dramatically
    smaller than the input.

    Args:
        encoder (EncoderBase): The encoder model to learn the latent space representation.
        decoder (DecoderBase): The decoder model to reconstruct the input from the latent
            space.
    
    > More information about the other arguments can be found in the super class.
    """
    def __init__(
        self,
        encoder: EncoderBase,
        decoder: DecoderBase,
        loss: str = "MSELoss",
        activation: str = "tanh",
        optimizer: str = "Adam",
        lr: float = 0.001,
        lr_decay: float = 0.999,
        monte_carlo: int = 0,
    ) -> None:
        super().__init__(
            loss=loss,
            activation=activation,
            optimizer=optimizer,
            lr=lr,
            lr_decay=lr_decay,
            monte_carlo=monte_carlo,
        )

        # build encoder/decoder
        self._encoder = encoder
        self._decoder = decoder

        self.save_hyperparameters(ignore=["encoder", "decoder"])

    def summary(self, meta: Metadata) -> str:
        samples = meta.num_input_sequence
        sensors = meta.num_features_sensors
        weather = meta.num_features_weather

        input_data = {
            "sensors": torch.tensor(
                np.zeros((1, samples, sensors), dtype=np.float32)
            ),
            "weather": torch.tensor(
                np.zeros((1, samples, weather), dtype=np.float32)
            ),
        }

        return str(torchsummary(self, input_data={"batch": input_data}))

    @property
    def target(self) -> str:
        """The target name of inputs to predict."""
        return self._decoder.output_keys

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Passes the input through both the encoder and decoder.

        Args:
            batch (Dict[str, torch.Tensor]): The input tensor.

        Returns:
            torch.Tensor: The output of the autoencoder.
        """
        x = self.encode(batch)
        x = self._decoder(x, batch)
        return self._activation(x)

    def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encodes the input using the encoder network.

        Args:
            inputs (Dict[str, torch.Tensor]): The input tensor.

        Returns:
            torch.Tensor: The encoded output.
        """
        keys = self._encoder.input_keys
        if len(keys) == 1:
            return self._encoder(inputs[keys[0]])

        x = inputs[keys[0]]
        for key in keys[1:]:
            x = torch.cat([x, inputs[key]], dim=-1)

        return self._encoder(x)

    def decode(self, z: torch.Tensor, inputs: torch.Tensor = None) -> torch.Tensor:
        """
        Decodes the input using the decoder network.

        Args:
            z (torch.Tensor): The input tensor to be decoded.
            inputs (torch.Tensor): The original input tensor, required when
                bypassing the encoder.

        Returns:
            torch.Tensor: The decoded output.
        """
        return self._decoder(z, inputs)
