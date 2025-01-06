"""This module implements a basic AutoEncoder class and the base classes for an
encoder and decoder."""
from typing import List

import torch
import torch.nn as nn

from Comprisk.model import BaseModel


class EncoderBase(nn.Module):
    """
    The encoder base class implements the forward and predict methods, as well as initializer
    function to define the encoder.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(EncoderBase, self).__init__()
        self.device = torch.device("cpu")

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
        self.device = torch.device("cpu")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Implements the forward pass of the decoder."""
        raise NotImplementedError()


class AutoEncoder(BaseModel):
    """The AutoEncoder class uses a given encoder and decoder model to learn a latent
    space representation of the input data. Usually, the latent space is dramatically
    smaller than the input.
    
    > More information about the other arguments can be found in the super class.
    """
    def __init__(
        self,
        features_in: List[str] = None,
        features_out: List[str] = None,
        loss: str = "MSELoss",
        activation: str = "tanh",
        optimizer: str = "Adam",
        lr: float = 0.001,
        lr_decay: float = 0.999,
    ) -> None:
        super().__init__(
            features_in=features_in,
            features_out=features_out,
            loss=loss,
            activation=activation,
            optimizer=optimizer,
            lr=lr,
            lr_decay=lr_decay,
        )

        # build encoder/decoder
        self._encoder = self.encoder_init()
        self._encoder.to(self.device)
        self._encoder.device = self.device

        self._decoder = self.decoder_init()
        self._decoder.to(self.device)
        self._encoder.device = self.device

        # self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.save_hyperparameters()

    def encoder_init(self) -> nn.Module:
        """This method must be overwritten by a child class to instantiate the encoder part of the autoencoder. This
        initialization method is called after the super-constructor of the BaseModel class was called.

        Returns:
            nn.Module: The Encoder as pytorch module.
        """
        raise NotImplementedError

    def decoder_init(self) -> nn.Module:
        """This method must be overwritten by a child class to instantiate the decoder part of the autoencoder. This
        initialization method is called after the super-constructor of the BaseModel class was called.

        Returns:
            nn.Module: The Encoder as pytorch module.
        """
        raise NotImplementedError

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
