import os.path
from typing import Union, Dict, Tuple, List

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch import nn
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
        trange: Tuple = (-1.0, 1.0),
        features_in: List[str] = None,
        features_out: List[str] = None,
    ) -> None:
        super(NormalizeModule, self).__init__()

        self._features_in = sorted(features_in)
        self._features_out = sorted(features_out)

        self._trange_min = Parameter(Tensor([trange[0]]))
        self._trange_max = Parameter(Tensor([trange[1]]))
        self._trange = Parameter(Tensor([trange[1] - trange[0]]))

        self._trained = False

        self._min: nn.ParameterDict | None = None
        self._scale: nn.ParameterDict | None = None

    @staticmethod
    def import_normalization(export: Dict) -> "NormalizeModule":
        normalizer = NormalizeModule(
            trange=(
                export["normalizer"]["trange"]["min"],
                export["normalizer"]["trange"]["max"],
            ),
            features_in=export["features"]["input"],
            features_out=export["features"]["output"],
        )

        min_dict = {}
        scale_dict = {}

        for key, feature_list in export["features"].items():
            min_ = np.array(
                [export["normalizer"][k]["min"] for k in sorted(feature_list)]
            )
            max_ = np.array(
                [export["normalizer"][k]["max"] for k in sorted(feature_list)]
            )

            min_dict[key] = Parameter(Tensor(min_))
            scale_dict[key] = Parameter(Tensor(1 / (max_ - min_)))

        normalizer._min = nn.ParameterDict(min_dict)
        normalizer._scale = nn.ParameterDict(scale_dict)
        normalizer._trained = True

        return normalizer

    def export_normalization(self) -> Dict:
        result = {
            "trange": {
                "min": float(self._trange_min.cpu().detach()),
                "max": float(self._trange_min.cpu().detach()),
            }
        }

        for i, feature in enumerate(self._features_in):
            result[feature] = {
                "min": float(self._min["input"][i].cpu().detach()),
                "scale": float(self._scale["input"][i].cpu().detach()),
            }

        return {
            "normalizer": result,
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

    def apply(self, x: BATCH_TYPE) -> BATCH_TYPE:
        x, y = x

        # normalize to 0 to 1
        x = (x - self._min["input"]) * self._scale["input"]
        y = (y - self._min["output"]) * self._scale["output"]

        # change range to -1 to 1
        x = (x + self._trange_min / self._trange) * self._trange
        y = (y + self._trange_min / self._trange) * self._trange
        return x, y

    def reverse(self, x: BATCH_TYPE) -> BATCH_TYPE:
        x, y = x

        # change range to -1 to 1
        x = (x / self._trange) - self._trange_min * self._trange
        y = (y / self._trange) - self._trange_min * self._trange

        # normalize to 0 to 1
        x = (x / self._scale["input"]) + self._min["output"]
        y = (y / self._scale["input"]) + self._min["output"]
        return x, y


class NormalizeWrapper:
    def __init__(self) -> None:
        self._normalizer: NormalizeModule | None = None

    @property
    def normalizer(self) -> NormalizeModule:
        return self._normalizer

    def normalization_import(
        self, export: Dict | None = None, path: str | None = None
    ) -> None:
        if path:
            path = os.path.join(path, "model.yaml")
            if os.path.exists(path):
                with open(path, "r") as f:
                    export = yaml.safe_load(f)

        if export:
            self._normalizer = NormalizeModule.import_normalization(export)

    def normalization_save(self, path: str | None = None) -> None:
        meta = {}
        path = os.path.join(path, "model.yaml")
        if os.path.exists(path):
            with open(path, "r") as f:
                meta = yaml.safe_load(f)

        export = self._normalizer.export_normalization()
        meta["normalizer"] = export["normalizer"]
        meta["features"] = export["features"]

        with open(path, "w") as f:
            yaml.safe_dump(meta, f)
