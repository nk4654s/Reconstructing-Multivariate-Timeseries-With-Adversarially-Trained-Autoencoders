import numpy as np
from tqdm import trange
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Comprisk.custom import DataModule as BaseModule
from Dataflux.cli import FluxConfigManager
from Dataflux.pipeline import pipeline as flux_pipeline
from Dataflux import get_logger

logger = get_logger(__name__)


class DataModule(BaseModule):
    def __init__(self, train: str | None = None, validate: str | None = None, standardize: bool = True, *args, **kwargs) -> None:
        train = flux_pipeline(FluxConfigManager(train))
        validate = flux_pipeline(FluxConfigManager(validate))

        super().__init__(*args, train=train, validate=validate, **kwargs)

        self.standardize = standardize
        self._scaler = StandardScaler() if self.standardize else MinMaxScaler()

    @property
    def scaler(self):
        return self._scaler

    def standardizer_params(self, output: bool = False) -> tuple[np.ndarray, np.ndarray]:
        assert self.scaler, "No scaler initialized to get parameters from"
        assert self.trainset, "No training dataset loaded to fit the standardizer to."

        s = self.trainset.get_args().stride
        self.trainset.set_stride(self.trainset.get_args().sequence_in)
        j = 1 if output else 0

        sn = len(self.trainset)
        le, f = self.trainset[0][j].shape
        logger.debug("Standardization on dataset shape (%s, %s, %s)", sn, le, f)

        if output:
            f = self.trainset.get_args().features_out

        data = np.zeros((sn * le, f))
        for i in trange(0, sn, desc="Fitting " + ("Standard Scaler" if self.standardize else "Normalizer")):
            data[i * le : (i + 1) * le] = self.trainset[i][j]

        self._scaler = self.scaler.fit(data)
        self.trainset.set_stride(s)
        if self.standardize:
            return self.scaler.mean_, self.scaler.scale_
        else:
            return self.scaler.min_, self.scaler.scale_
