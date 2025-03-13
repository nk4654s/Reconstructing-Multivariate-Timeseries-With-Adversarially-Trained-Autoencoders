import numpy as np
import pandas as pd
from torch import nn, Tensor

from Compmesh.model import ScatteredGridInterpolator
from Comprisk.model import BaseModel


def process_df(data: pd.DataFrame, label: dict[str, str]) -> tuple[Tensor, list[str], dict]:
    sub = data[label.keys()].copy()
    sub.rename(columns=label, inplace=True)
    sub = sub.reindex(sorted(sub.columns), axis=1)

    norm_data = {key: {"min": sub[key].min(), "max": sub[key].max()} for key in sub.columns}
    return Tensor(sub.values), sub.columns, norm_data


class SupportPointModel(BaseModel):
    def __init__(self, data: pd.DataFrame, x_label: dict[str, str], y_label: dict[str, str]):
        """
        A model that first interpolates, then applies a learnable linear transformation.
        Args:
            data (pd.DataFrame): The support points loaded as DataFrame.
            x_label: (dict[str, str]): The name mapping of input vars.
            x_label: (dict[str, str]): The name mapping of output vars.
        """
        # process the support points and initialize the normalization layer
        gx, x_label, x_norm = process_df(data, x_label)
        gy, y_label, y_norm = process_df(data, y_label)
        super().__init__(features_in=list(x_norm.keys()), features_out=list(y_norm.keys()))

        self.init_normalization(x_norm, y_norm)

        # build the grid interpolation and linear transformation parameters
        gx, gy = self.normalizer.apply((gx, gy))
        self.interpolator = ScatteredGridInterpolator(gx, gy)
        self.slope = nn.Parameter(Tensor(np.zeros(gy.shape[1])))
        self.intercept = nn.Parameter(Tensor(np.zeros(gy.shape[1])))

    def init_normalization(self, x: dict, y: dict):
        dx = x.copy()
        dx.update(y)
        data = {
            "data": dx,
            "features": {"input": list(x.keys()), "output": list(y.keys())},
            "trange": {"min": 0, "max": 1},
        }

        self.normalization_import(data)

    def forward(self, x: Tensor):
        y_interp = self.interpolator(x)
        y_final = y_interp + self.intercept
        return y_final

    def predict_step(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        x, y = self.normalizer.apply((x, y))
        return self(x), y
