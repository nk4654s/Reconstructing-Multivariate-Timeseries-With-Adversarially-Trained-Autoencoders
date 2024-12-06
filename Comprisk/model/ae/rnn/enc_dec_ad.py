from typing import Tuple
import torch
import torch.nn as nn
from Comprisk.model.ae import EncoderBase, DecoderBase, AutoEncoder


# https://arxiv.org/pdf/1607.00148.pdf
class EncDecAD(AutoEncoder):
    def __init__(self, 
        encoder: EncoderBase,
        decoder: DecoderBase,
        loss: str = "MSELoss",
        activation: str = "tanh",
        optimizer: str = "Adam",
        lr: float = 0.001,
        lr_decay: float = 0.999
    ) -> None:
        super(EncDecAD, self).__init__(encoder, decoder, loss, activation, optimizer, lr, lr_decay)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(size=batch.shape, dtype=torch.float, device=self.device)
        hidden_cell = self._encoder(batch)

        input_decoder = batch[:, -1, :].view(batch.shape[0], 1, batch.shape[2])
        for i in range(batch.shape[1] - 1, -1, -1):
            output_decoder, hidden_cell = self._decoder(input_decoder, hidden_cell)
            input_decoder = output_decoder
            output[:, i, :] = output_decoder[:, 0, :]

        return output


class EDAEncoder(EncoderBase):
    def __init__(self, num_hidden_layers: int, num_hidden_dim: int, num_features: int, dropout_prob: float = 0.0):
        super(EDAEncoder, self).__init__()

        self.input_size = num_features
        self.hidden_size = num_hidden_dim
        self.num_layers = num_hidden_layers

        self.lstm = nn.LSTM(input_size=num_features, hidden_size=num_hidden_dim,
                            num_layers=num_hidden_layers, batch_first=True,
                            dropout=dropout_prob, bias=True, device=self.device)

    def initHidden(self, batch_size):
        self.hidden_cell = (
            torch.randn((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device),
            torch.randn((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device)
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        _, self.hidden_cell = self.lstm(batch)
        return self.hidden_cell


class EDADecoder(DecoderBase):
    def __init__(self, num_hidden_layers: int, num_hidden_dim: int, num_features: int, dropout_prob: float = 0.0):
        super(EDADecoder, self).__init__()

        self.input_size = num_features
        self.hidden_size = num_hidden_dim
        self.num_layers = num_hidden_layers

        self.lstm = nn.LSTM(input_size=num_features, hidden_size=num_hidden_dim,
                            num_layers=num_hidden_layers, batch_first=True,
                            dropout=dropout_prob, bias=True, device=self.device)
        self.linear = nn.Linear(in_features=num_hidden_dim, out_features=num_features, device=self.device)

    def forward(self, batch: torch.Tensor, hidden_cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, hidden_cell = self.lstm(batch, hidden_cell)
        output = self.linear(output)
        return output, hidden_cell