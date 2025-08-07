import torch

from CompriskV2.model import BaseModel

class GANClassificator(torch.nn.Module):
    """
    simple num_features to 1 classificator with 3 dense layers for now
    """


    def __init__(
        self,
        num_features: int,
        sequence_input_size: int,
        #num_layers: int = 1,
    )-> None:

        super().__init__()
        self._num_features = num_features
        self._sequence_input_size = sequence_input_size
        self._activation = torch.nn.LeakyReLU()

        self._conv_layer1 = torch.nn.Conv1d(
            in_channels=self._num_features,
            out_channels=self._num_features * 4,
            kernel_size=3,
            padding=1,
        )  # 60 * 88

        self._pool1 = torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=1)  # 30 * 88

        self._conv_layer2 = torch.nn.Conv1d(
            in_channels=self._num_features * 4,
            out_channels=self._num_features * 8,
            kernel_size=3,
            padding=1,
        )  # 30 * 176

        self._pool2 = torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=1) # 15 * 176

        self._conv_layer3 = torch.nn.Conv1d(
            in_channels=self._num_features * 8,
            out_channels=self._num_features,
            kernel_size=3,
            padding=1,
        )  # 15 * 22



    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = torch.swapaxes(batch, -2, -1)

        output = self._conv_layer1(batch)
        output = self._activation(output)
        output = self._pool1(output)

        output = self._conv_layer2(output)
        output = self._activation(output)
        output = self._pool2(output)

        output = self._conv_layer3(output)
        output = self._activation(output)
        output = torch.swapaxes(output, -2, -1)
        #output = torch.nn.Sigmoid()(output)

        return output
