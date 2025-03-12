import torch
from Dataflux import logging_mode
from Dataflux.logger import get_logger
from torch import nn

from Comprisk.model import BaseModel

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class MultimodalForecast(BaseModel):
    def __init__(self, dropout_prob: float = 0.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._sequence_size = 600
        self._future_size = 60
        self._kernel_size = 31
        self._padding = 15
        self._latent_size = self._input_dim // 2
        self._num_layers = 4
        self._stride = 2
        self._dropout = dropout_prob

        if self._future_size >= self._sequence_size:
            raise ValueError(
                "The 'future_size' must be smaller than the 'sequence_size' (%s<%s)",
                self._sequence_size,
                self._future_size,
            )

        layers = []

        # calculate factors
        f_1 = (self._latent_size / self._input_dim) ** (1 / self._num_layers)

        # calculate dimensions
        feature_dim = self._input_dim
        sequence_dim = self._sequence_size
        forward_feature_dims = []
        for n in range(0, self._num_layers + 1):
            feature_dim = round(f_1**n * self._input_dim)
            forward_feature_dims.append(feature_dim)

            if n > 0:
                sequence_dim = ((sequence_dim - self._kernel_size + 2 * self._padding) // self._stride) + 1

        logger.debug("Model forward dimension goes: %s", forward_feature_dims)

        if forward_feature_dims[-1] == 0:
            raise ValueError(
                "The latent space of the last block is 0. Increase the latent factor or reduce number of layers."
            )

        f_2 = (self._output_dim / feature_dim) ** (1 / self._num_layers)
        f_3 = (self._future_size / sequence_dim) ** (1 / self._num_layers)
        logger.debug("Factors f1=%s, f2=%s, f3=%s", f_1, f_2, f_3)

        future_feature_dims = []
        for n in range(0, self._num_layers + 1):
            feature_dim = round(f_2**n * forward_feature_dims[-1])
            future_feature_dims.append(feature_dim)

            if n > 0:
                sequence_dim = ((sequence_dim - self._kernel_size + 2 * self._padding) // 1) + 1

        logger.debug("Model future dimension goes: %s", future_feature_dims)
        logger.info("Model outputs a sequence of %s samples", sequence_dim)

        for i in range(self._num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        forward_feature_dims[i],
                        forward_feature_dims[i + 1],
                        self._kernel_size,
                        self._stride,
                        padding=self._padding,
                        device=self._device,
                    ),
                    nn.BatchNorm1d(forward_feature_dims[i + 1], device=self._device),
                    nn.Dropout(self._dropout),
                    nn.GELU(),
                )
            )

        for i in range(self._num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        future_feature_dims[i],
                        future_feature_dims[i + 1],
                        self._kernel_size // 2,
                        1,
                        padding=self._padding,
                        device=self._device,
                    ),
                    nn.BatchNorm1d(future_feature_dims[i + 1], device=self._device),
                    nn.Dropout(self._dropout),
                    nn.GELU(), # if i != self._num_layers - 1 else self._activation
                )
            )

        self._blocks = nn.ModuleList(layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.swapaxes(1, 2)
        for block in self._blocks:
            x = block(x)

        return x.swapaxes(1, 2)
