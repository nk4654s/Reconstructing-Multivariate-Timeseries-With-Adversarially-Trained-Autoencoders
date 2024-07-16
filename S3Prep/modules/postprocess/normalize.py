from threading import Lock
from typing import Any, List
import pandas as pd
from S3Prep.helper import Meta, Metadata
from S3Prep.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency

mutex = Lock()


class TrainNormalize(FileWorkerCallable):
    def __init__(self):
        self._min, self._max = None, None

    def _fit(self, _min, _max) -> None:
        if self._min is None:
            self._min = _min
        else:
            self._min[_min < self._min] = _min[_min < self._min]

        if self._max is None:
            self._max = _max
        else:
            self._max[_max > self._max] = _max[_max > self._max]

    def get_dependencies(self) -> List[str]:
        return {ApplyNormalize.__name__: CallableDependency.IMMEDIATELY_AFTER}

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        # Load the file into a pandas DataFrame
        df = pd.read_hdf(file_path, key="data")

        # Calculate minimum and maximum per column
        df_min = df.min().to_numpy()
        df_max = df.max().to_numpy()
        with mutex:
            self._fit(df_min, df_max)

            # pass the minimum and maximum to the next callable
            context.put(TrainNormalize.__name__, (self._min, self._max))

    def finish(self, context: FileWorkerContext, _: Metadata) -> None:
        pass


class ApplyNormalize(FileWorkerCallable):
    def __init__(self, ignore_column: str = "ts") -> None:
        self._c = ignore_column

    def get_dependencies(self) -> List[str]:
        return {TrainNormalize.__name__: CallableDependency.IMMEDIATELY_BEFORE}

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        a, b = context.get(TrainNormalize.__name__)
        df = pd.read_hdf(file_path, key="data")

        normed = (df - a).div(b - a).fillna(0)
        df.T[df.columns.values != self._c] = normed.T[df.columns.values != self._c]
        df.to_hdf(file_path, key="norm")

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        a, b = context.get(TrainNormalize.__name__)
        context.clear(TrainNormalize.__name__)

        data = {
            Meta.NORM_METHOD.name: "MinMaxNormalizer",
            Meta.NORM_VAL_A.name: list(a),
            Meta.NORM_VAL_B.name: list(b)
        }
        metadata.put(data, Meta.NORMALIZER)
