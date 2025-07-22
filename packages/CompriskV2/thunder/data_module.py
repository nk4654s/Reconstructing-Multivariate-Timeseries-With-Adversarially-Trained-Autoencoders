import numpy as np
from tqdm import trange
from sklearn.preprocessing import StandardScaler

from Comprisk.custom import DataModule as BaseModule
from Dataflux.cli import FluxConfigManager
from Dataflux.pipeline import pipeline as flux_pipeline


class DataModule(BaseModule):
    def __init__(self, train: str | None = None, validate: str | None = None, *args, **kwargs) -> None:
        train = flux_pipeline(FluxConfigManager(train))
        validate = flux_pipeline(FluxConfigManager(validate))

        super().__init__(*args, train=train, validate=validate, **kwargs)

        self._scaler = StandardScaler()

    @property
    def scaler(self):
        return self._scaler

    def standardizer_params(self, output: bool = False) -> tuple[np.ndarray, np.ndarray]:
        assert self.scaler, "No scaler initialized to get parameters from"
        assert self.trainset, "No training dataset loaded to fit the standardizer to."

        s = self.trainset.get_args().stride
        self.trainset.set_stride(self.trainset.get_args().sequence_in)
        sn, le, f = self.trainset.shape

        if output:
            f = self.trainset.get_args().features_out

        data = np.zeros((sn * le, f))
        j = 1 if output else 0
        for i in trange(0, sn, desc="Fitting Standard Scaler"):
            data[i * le : (i + 1) * le] = self.trainset[i][j]

        self._scaler = self.scaler.fit(data)
        self.trainset.set_stride(s)
        return self.scaler.mean_, self.scaler.scale_
