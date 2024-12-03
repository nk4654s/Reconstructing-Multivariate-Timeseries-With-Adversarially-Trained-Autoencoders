"""This module implements a basic AutoEncoder class and the base classes for an
encoder and decoder."""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary as torchsummary

from Dataflux.pytorch.metadata import Metadata
from Comprisk.model import BaseModel
from Comprisk import logger


class EncoderBase(nn.Module):
    """
    The encoder base class implements the forward and predict methods, as well as initializer
    function to define the encoder.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(EncoderBase, self).__init__()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Implements the forward pass of the encoder."""
        raise NotImplementedError()


class DecoderBase(nn.Module):
    """
    The decoder base class implements the forward and predict methods, as well as initializer
    function to define the decoder.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(DecoderBase, self).__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Implements the forward pass of the decoder."""
        raise NotImplementedError()


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
    ) -> None:
        super().__init__(
            loss=loss,
            activation=activation,
            optimizer=optimizer,
            lr=lr,
            lr_decay=lr_decay,
        )

        # build encoder/decoder
        self._encoder = encoder
        self._decoder = decoder

        self.save_hyperparameters(ignore=["encoder", "decoder"])

    @property
    def target(self) -> str:
        """The target name of inputs to predict."""
        return self._decoder.output_keys

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through both the encoder and decoder.

        Args:
            batch (Dict[str, torch.Tensor]): The input tensor.

        Returns:
            torch.Tensor: The output of the autoencoder.
        """
        x = self.encode(batch)
        x = self._decoder(x)
        return self._activation(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input using the encoder network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The encoded output.
        """
        x = self._encoder(x)
        return torch.tanh(x)

    def decode(self, z: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
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
