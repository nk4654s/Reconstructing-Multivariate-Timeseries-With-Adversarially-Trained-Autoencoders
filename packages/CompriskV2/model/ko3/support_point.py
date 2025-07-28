import os
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
        interpolator: str = "rbf",
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
        assert interpolator in [
            "rbf",
            "grid",
        ], "Allowed interpolator is either 'rbf' or 'grid'"
        super().__init__(*args, **kwargs)

        self._ko3_table = ko3_table

        self._interpolator_cls = RBFGridInterpolator
        if interpolator != "rbf":
            self._interpolator_cls = ScatteredGridInterpolator

        self.interpolator: ScatteredGridInterpolator | None = None
        self.calibrator: LinearCalibration | None = None

    def on_validation_start(self) -> None:
        if self.interpolator is not None:
            return

        # load ko3 files from disk
        mapping = os.path.join(self._ko3_table, "mapping.yaml")
        x_mapping, y_mapping = load_name_mapping(mapping)
        df = load_support_points_table(self._ko3_table)

        # process ko3 table
        gx, (mins, maxs) = process_df(df, x_mapping)
        self._input_transformer.set_params(mins, maxs - mins)
        logger.debug("KO3 table takes %s values")

        gy, (mins, maxs) = process_df(df, y_mapping)
        self._output_transformer.set_params(mins, maxs - mins)
        logger.debug("KO3 table outputs %s values")

        self.interpolator = self._interpolator_cls(gx[0].data, gy[0].data)
        self.calibrator = LinearCalibration(gy.shape[-1])

        self.interpolator.to(self.device)
        self.calibrator.to(self.device)

    @property
    def labels(self) -> tuple[str, ...]:
        return ("y_scaled", "y_hat", "score")

    def forward(
        self, x: Tensor, y: Tensor, return_score: bool = True
    ) -> tuple[Tensor, ...]:
        x, y = super().forward(x, y)

        if return_score:
            pass

        y_interp = self.interpolator(x)
        y_calib = self.calibrator(y_interp)
        return y, y_calib
