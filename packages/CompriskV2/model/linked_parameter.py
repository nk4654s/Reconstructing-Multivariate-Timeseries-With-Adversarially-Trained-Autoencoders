import numpy as np
import torch
from torch import Tensor, nn, zeros

from CompriskV2.model.base import BaseModel


class LinkedParameterModel(BaseModel):
    """This class links the dataset's input/output features and sequences to the model's
    hyperparameters. The linking happens in `CompriskV2.thunder.cli`. Furthermore, this
    class implements standardization in its `forward()` method. This enables exporting
    the standardization to `ONNX` as all operations are tracable, while the gradient is
    disabled."""

    def __init__(
        self,
        features_in: list[str],
        features_out: list[str],
        sequence_in: int,
        sequence_out: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._features_in = len(features_in)
        self._features_out = len(features_out)

        features_in = sorted(features_in)
        features_out = sorted(features_out)
        self._features_out_indicies = [features_in.index(_) for _ in features_out]

        self._sequence_in = sequence_in
        self._sequence_out = sequence_out

        self._mean = nn.Parameter(zeros(self._features_in))
        self._scale = nn.Parameter(zeros(self._features_in))

        self._eps = 1e-6

    @property
    def embedding(self) -> tuple[Tensor, Tensor]:
        return (
            zeros((1, self._sequence_in, self._features_in)),
            zeros((1, self._sequence_out, self._features_out)),
        )

    @property
    def labels(self) -> tuple[str]:
        return ("x", "y")

    @torch.no_grad()
    def forward(self, x: Tensor, y: Tensor, **kwargs) -> tuple[Tensor, ...]:
        x = (x - self._mean) / (self._scale + self._eps)
        y = (y - self._mean[self._features_out_indicies]) / (
            self._scale[self._features_out_indicies] + self._eps
        )
        return x, y

    def set_standardizer(self, mean: np.ndarray, scale: np.ndarray) -> None:
        """This method sets the standardization parameters (e.g. gathered from sklearn's Standardizer).

        Args:
            mean (np.ndarray): The mean values of the data distribution.
            scal (np.ndarray): The distribution's scale.
        """
        self._mean = nn.Parameter(torch.Tensor(mean))
        self._scale = nn.Parameter(torch.Tensor(scale))
