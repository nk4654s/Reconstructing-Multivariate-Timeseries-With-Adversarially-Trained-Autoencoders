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
        a, b = Tensor(a), Tensor(b)
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
            b = b.unsqueeze(0)

        self._a = nn.Parameter(a)
        self._b = nn.Parameter(b)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return (x - self._a) / (self._b + self._eps)


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
        features_in = sorted(features_in)
        features_out = sorted(features_out)

        self._names_in = features_in
        self._names_out = features_out

        self._features_in = len(features_in)
        self._features_out = len(features_out)

        self._sequence_in = max(sequence_in, 1)
        self._sequence_out = max(sequence_out, 1)

        self._input_transformer = SampleTransformer(self._features_in)
        self._output_transformer = SampleTransformer(self._features_out)

    @property
    def embedding(self) -> tuple[Tensor, Tensor]:
        return (
            zeros((1, self._sequence_in, self._features_in)),
            zeros((1, self._sequence_out, self._features_out)),
        )

    @property
    def require_standardization(self) -> bool:
        return True

    @property
    def labels(self) -> tuple[str]:
        return ("x", "y")

    @torch.no_grad()
    def forward(self, x: Tensor, y: Tensor, **kwargs) -> tuple[Tensor, ...]:
        self._input_transformer.to(x.device)
        self._output_transformer.to(y.device)
        return (
            self._input_transformer(x),
            self._output_transformer(y),
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
