from typing import Union, Dict, Tuple, List, Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch import nn
from lightning import LightningModule
from tqdm import tqdm

from Dataflux.pytorch import Dataset, BATCH_TYPE
from Dataflux import get_logger, logging_mode


logger = get_logger(__name__, "logfile", logging_mode)
PREDICT_TYPE = Union[np.ndarray, torch.Tensor]


class ModelNotTrainedError(BaseException):
    pass


class NormalizeModule(nn.Module):
    def __init__(
        self,
        features_in: List[str],
        features_out: List[str],
        trange: Tuple = (-1.0, 1.0),
        normalize_data: Dict = None,
    ) -> None:
        super().__init__()

        self._features_in = sorted(features_in)
        self._features_out = sorted(features_out)

        self._input_idx = [i for i, n in enumerate(self._features_in)]
        self._output_idx = [self._features_in.index(n) for n in self._features_out]

        self._trained = False

        self._min: nn.Parameter = Parameter(Tensor(np.zeros((len(self._features_in),))))
        self._scale: nn.Parameter = Parameter(Tensor(np.zeros((len(self._features_in),))))

        self._init_params(normalize_data)

        self._trange_min = Parameter(Tensor([trange[0]]))
        self._trange_max = Parameter(Tensor([trange[1]]))
        self._trange = Parameter(Tensor([trange[1] - trange[0]]))

    def _init_params(self, data: Dict) -> None:
        if not data:
            return None

        min_ = np.array([data[k]["min"] for k in self._features_in]).round(6)
        max_ = np.array([data[k]["max"] for k in self._features_in]).round(6)
        scale_ = (1 / (max_ - min_)).round(6)

        self._min = Parameter(Tensor(min_))
        self._scale = Parameter(Tensor(scale_))
        self._trained = True

    @staticmethod
    def import_normalization(export: Dict) -> "NormalizeModule":
        normalizer = NormalizeModule(
            trange=(
                export["trange"]["min"],
                export["trange"]["max"],
            ),
            features_in=export["features"]["input"],
            features_out=export["features"]["output"],
            normalize_data=export["data"],
        )

        return normalizer

    def export_normalization(self) -> Dict:
        trange = {
            "min": float(self._trange_min.cpu().detach()),
            "max": float(self._trange_max.cpu().detach()),
        }

        data = {}
        for i, feature in enumerate(self._features_in):
            min_ = round(float(self._min[self._input_idx][i].cpu().detach()), 6)
            scale_ = float(self._scale[self._input_idx][i].cpu().detach())
            max_ = round((1 / scale_) + min_, 6)
            data[feature] = {
                "min": min_,
                "max": max_,
            }

        return {
            "trange": trange,
            "data": data,
            "features": {"input": self._features_in, "output": self._features_out},
        }

    @torch.no_grad()
    def fit(self, data: DataLoader, verbose: bool = False) -> None:
        dataset: Dataset = data.dataset
        dataset.set_stride(60)

        # find the min and max values of the data
        mins: np.ndarray | None = dataset[0][0]
        maxs: np.ndarray | None = dataset[0][0]
        for x, y in tqdm(data, desc="Normalize Wrapper", disable=not verbose):
            mins_ = np.min(x, axis=0)
            mins[mins_ < mins] = mins_

            maxs_ = np.max(x, axis=0)
            mins[maxs_ > maxs] = maxs_

        # the min and scale must be a parameter of this model such that it will be
        # visible in the model graph. This is required by ONNX
        self._min = Parameter(Tensor(mins))
        self._scale = Parameter(Tensor(1 / (maxs - mins)))
        self._trained = True

        logger.debug("Trained %s successfully", self.__name__)
        logger.debug("Minimum values %s", mins)
        logger.debug("Maximum values %s", maxs)

    @torch.no_grad()
    def apply(self, x: BATCH_TYPE) -> BATCH_TYPE:
        x, y = x

        # normalize to 0 to 1
        x = (x - self._min[self._input_idx]) * self._scale[self._input_idx]
        y = (y - self._min[self._output_idx]) * self._scale[self._output_idx]

        # change range to -1 to 1
        x = (x + self._trange_min / self._trange) * self._trange
        y = (y + self._trange_min / self._trange) * self._trange

        return x, y

    @torch.no_grad()
    def reverse(self, x: Tensor | None, y: Tensor | None) -> BATCH_TYPE:
        if x is not None:
            x = (x / self._trange) - self._trange_min / self._trange
            x = (x / self._scale[self._input_idx]) + self._min[self._input_idx]

        if y is not None:
            y = (y / self._trange) - self._trange_min / self._trange
            y = (y / self._scale[self._output_idx]) + self._min[self._output_idx]

        return x, y


class NormalizeWrapper(LightningModule):
    def __init__(
        self,
        features_in: List[str] = None,
        features_out: List[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._normalizer = NormalizeModule(features_in=features_in, features_out=features_out)

    @property
    def normalizer(self) -> NormalizeModule:
        return self._normalizer

    def normalization_import(self, export: Dict) -> None:
        self._normalizer = NormalizeModule.import_normalization(export)
        self._normalizer.to(self.device)
