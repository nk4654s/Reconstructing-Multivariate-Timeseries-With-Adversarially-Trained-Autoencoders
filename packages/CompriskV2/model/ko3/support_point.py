import os
import torch
from torch import Tensor

from CompriskV2.model.ko3.utils import (
    load_name_mapping,
    load_support_points_table,
    process_df,
)
from CompriskV2.model.ko3 import (
    ScatteredGridInterpolator,
    RBFGridInterpolator,
    LinearCalibration,
    ScoreModule,
)
from CompriskV2.model import LinkedParameterModel

from Dataflux import get_logger

logger = get_logger(__name__)


class SupportPointModel(LinkedParameterModel):
    """The SupportPointModel loads a grid exported by KO3 to perform a prediction of sensors given an input. This model
    uses the RBFGridInterpolator to build a surrogate model and to sample coordinates from inbetween cells. Then, a
    linear transformation is applied to the surrogate's output as the exported grid from KO3 may require a calibration
    to work properly.
    """

    def __init__(
        self,
        ko3_table: str = "",
        kernel: str = "gaussian",
        *args,
        **kwargs,
    ):
        """
        A model that first interpolates, then applies a learnable linear transformation.
        Args:
            data (pd.DataFrame): The support points loaded as DataFrame.
            x_label: (dict[str, str]): The name mapping of input vars.
            x_label: (dict[str, str]): The name mapping of output vars.
        """
        super().__init__(*args, **kwargs)

        self._ko3_table = ko3_table
        self._score = ScoreModule(dim=self._features_out)

        # load ko3 files from disk
        mapping = os.path.join(self._ko3_table, "mapping.yaml")
        x_mapping, y_mapping = load_name_mapping(mapping)
        df = load_support_points_table(self._ko3_table)

        # process ko3 table, re-define transformers
        gx, (mins, maxs) = process_df(df, x_mapping)
        self._input_transformer.set_params(mins, maxs - mins)
        logger.debug("KO3 table takes %s values", len(mins))

        gy, (mins, maxs) = process_df(df, y_mapping)
        self._output_transformer.set_params(mins, maxs - mins)
        logger.debug("KO3 table outputs %s values", len(mins))

        # define interpolator and calibrator
        self.interpolator = RBFGridInterpolator(gx[0].data, gy[0].data, kernel)
        self.calibrator = LinearCalibration(gy.shape[-1])

    @property
    def require_standardization(self) -> bool:
        return False

    @property
    def onnx_name(self) -> str:
        return "support"

    def on_train_start(self) -> None:
        self.interpolator.to(self.device)
        self.calibrator.to(self.device)
        self._score.setup()

    def on_validation_end(self) -> None:
        self._score.finalize(self.device)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        y, y_pred = self(*batch, return_score=False)
        self._score.record(y)

        loss = self._loss(y, y_pred)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        y, y_pred, score = self(*batch)
        self._score.record(y)

        loss = self._loss(y, y_pred)
        self.log("val/loss", loss)
        self.log("val/score", torch.mean(score))
        return loss

    @property
    def labels(self) -> tuple[str, ...]:
        return ("y_scaled", "y_hat", "score")

    def forward(
        self, x: Tensor, y: Tensor, return_score: bool = True
    ) -> tuple[Tensor, ...]:
        x, y = super().forward(x, y)

        y_interp = self.interpolator(x)
        y_calib = self.calibrator(y_interp)

        if return_score:
            return y, y_calib, self._score(y, y_calib)

        return y, y_calib
