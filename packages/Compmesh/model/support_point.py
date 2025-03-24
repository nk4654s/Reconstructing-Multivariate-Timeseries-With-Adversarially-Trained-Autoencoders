import pandas as pd
from torch import Tensor

from Compmesh.model import ScatteredGridInterpolator, RBFGridInterpolator
from Compmesh.model.calibration import LinearCalibration
from Comprisk.model import BaseModel


def process_df(data: pd.DataFrame, label: dict[str, str]) -> tuple[Tensor, list[str], dict]:
    sub = data[label.keys()].copy()
    sub.rename(columns=label, inplace=True)
    sub = sub.reindex(sorted(sub.columns), axis=1)

    norm_data = {key: {"min": sub[key].min(), "max": sub[key].max()} for key in sub.columns}
    return Tensor([sub.values]), sub.columns, norm_data


class SupportPointModel(BaseModel):
    """The SupportPointModel loads a grid exported by KO3 to perform a prediction of sensors given an input. This model
    uses the RBFGridInterpolator to build a surrogate model and to sample coordinates from inbetween cells. Then, a
    linear transformation is applied to the surrogate's output as the exported grid from KO3 may require a calibration
    to work properly.
    """
    def __init__(self, *args, **kwargs):
        """
        A model that first interpolates, then applies a learnable linear transformation.
        Args:
            data (pd.DataFrame): The support points loaded as DataFrame.
            x_label: (dict[str, str]): The name mapping of input vars.
            x_label: (dict[str, str]): The name mapping of output vars.
        """
        super().__init__(*args, **kwargs)

        self.interpolator: ScatteredGridInterpolator | None = None
        self.calibrator: LinearCalibration | None = None

    def init_grid(self, data: pd.DataFrame, x_label: dict[str, str], y_label: dict[str, str]):
        """This method loads the KO3 grid loaded as dataframe.

        Args:
            data (pd.DataFrame): The KO3 export.
            x_label (dict): The mapping of input values where the key corresponds to the value name of KO3 and the
                value represents the UUID used by Xplore.
            y_label (dict): Equal to `x_label` but maps the output values.
        """
        gx, x_label, x_norm = process_df(data, x_label)
        gy, y_label, y_norm = process_df(data, y_label)

        self.init_normalization(x_norm, y_norm)
        gx, gy = self.normalizer.apply((gx, gy))

        self.interpolator = RBFGridInterpolator(gx[0].data, gy[0].data)
        self.calibrator = LinearCalibration(gy.shape[-1])

        self.interpolator.to(self.device)
        self.calibrator.to(self.device)

    def init_normalization(self, x: dict, y: dict):
        """This method initializes the normalization layer.

        > Note: This should be called by this class automatically after initializing the KO3 grid.

        Args:
            x (dict): The input values in the shape {"name": value}.
            y (dict): Same as for `x`.
        """
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
        y_calib = self.calibrator(y_interp)
        return y_calib

    def predict_step(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        x, y = self.normalizer.apply((x, y))
        return self(x), y
