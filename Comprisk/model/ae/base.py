"""This module implements a base AutoEncoder class."""
import torch
from torch import nn

class AutoEncoder(nn.Module):
    """
    A simple autoencoder class.

    Attributes:
        encoder (nn.Module): The encoder network.
        decoder (nn.Module): The decoder network.
        loss_fn (callable): The loss function to use during training.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, loss_fn: callable):
        """
        Initializes the autoencoder.

        Args:
            encoder (nn.Module): The encoder network.
            decoder (nn.Module): The decoder network.
            loss_fn (callable): The loss function to use during training.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through both the encoder and decoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the autoencoder.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input using the encoder network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The encoded output.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the input using the decoder network.

        Args:
            z (torch.Tensor): The input tensor to be decoded.

        Returns:
            torch.Tensor: The decoded output.
        """
        return self.decoder(z)
