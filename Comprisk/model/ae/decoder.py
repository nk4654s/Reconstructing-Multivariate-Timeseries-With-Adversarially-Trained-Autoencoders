"""Simple Decoder model."""
import torch
from torch import nn, F


class Decoder(nn.Module):
    """
    A basic decoder network using a Convolutional Neural Network (CNN).

    Attributes:
        latent_space_size (int): The size of the latent space.
        num_filters (int): The number of filters used in the convolutional layers.
        dropout_prob (float): The probability of dropout used in the network.
        num_features (int): The number of features in the output data.
        kernel_size (int): The size of the convolutional kernels.
    """

    def __init__(
        self,
        latent_space_size: int = 10,
        num_filters: int = 64,
        dropout_prob: float = 0.2,
        num_features: int = 10,
        kernel_size: int = 3,
    ):
        """
        Initializes the decoder network.

        Args:
            latent_space_size (int): The size of the latent space. Defaults to 10.
            num_filters (int): The number of filters used in the convolutional layers.
            Defaults to 64.
            dropout_prob (float): The probability of dropout used in the network. Defaults to 0.2.
            num_features (int): The number of features in the output data. Defaults to 10.
            kernel_size (int): The size of the convolutional kernels. Defaults to 3.
        """
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(
            latent_space_size, num_filters, kernel_size=(kernel_size, latent_space_size)
        )
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters, num_features, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the decoder network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The decoded output.
        """
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        return x

    def calculate_output_size(self) -> int:
        """
        Calculates the size of the output.

        Returns:
            int: The size of the output.
        """
        return self.conv2.out_channels * (self.conv2.kernel_size[0] - 1)
