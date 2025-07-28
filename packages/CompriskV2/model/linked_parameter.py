import numpy as np
import torch
from torch import Tensor, nn, zeros

from CompriskV2.model.base import BaseModel
from Dataflux import get_logger

logger = get_logger(__name__)


class SampleTransformer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        logger.debug("Init transfomer module with %s values", dim)
        self._a = nn.Parameter(zeros(dim))
        self._b = nn.Parameter(zeros(dim))

        self._eps = 1e-6

    def set_params(self, a: np.array, b: np.array) -> None:
        self._a = nn.Parameter(Tensor(a))
        self._b = nn.Parameter(Tensor(b))

    def standardize(self, x: Tensor) -> Tensor:
        return (x - self._a) / (self._b + self._eps)

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self._b) / (self._a - self._b)


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
        logger.debug(
            "Init module with y=%s and y=%s", self._features_in, self._features_out
        )

        features_in = sorted(features_in)
        features_out = sorted(features_out)

        self._sequence_in = sequence_in
        self._sequence_out = sequence_out

        self._input_transformer = SampleTransformer(self._features_in)
        self._output_transformer = SampleTransformer(self._features_out)

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
        return (
            self._input_transformer.standardize(x),
            self._output_transformer.standardize(y),
        )

    def set_transformers(
        self,
        inputs: tuple[np.ndarray, np.ndarray],
        outputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """This method sets the standardization parameters (e.g. gathered from sklearn's Standardizer).

        Args:
            inputs (tuple): The input sample transformer params.
            outputs (tuple): The output sample transformer params.
        """
        self._input_transformer.set_params(*inputs)
        self._output_transformer.set_params(*outputs)
