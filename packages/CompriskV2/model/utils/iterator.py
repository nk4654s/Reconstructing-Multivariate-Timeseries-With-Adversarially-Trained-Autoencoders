"""This module contains helper methods related to the models."""

from dataclasses import InitVar, dataclass, field


from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import onnxruntime
import pandas as pd

from Dataflux.pytorch import collate_fn, Dataset
from Dataflux.logger import get_logger


logger = get_logger(__name__)


@dataclass
class SensorResult:
    data: dict[str, np.ndarray]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)


@dataclass
class IteratorResult:
    """This dataclass holds a model's iteration result including the scores, model inputs, and model outputs.
    Furthermore, this class is able to convert the result to pandas DataFrames.
    """

    names: list[str]
    keys: list[str]
    ts: int

    prediction: InitVar[dict[str, np.ndarray]]
    sensors: list[SensorResult] = field(default_factory=list)

    def __post_init__(self, prediction: dict[str, np.ndarray]):
        for i in range(len(self.names)):
            result = SensorResult(data={n: d[:, i] for n, d in prediction.items()})
            self.sensors.append(result)

    def by_name(self, name: str) -> SensorResult:
        if name not in self.names:
            raise ValueError("Name could not be found.")

        i = self.names.index(name)
        return self.sensors[i]

    def revert_with(
        self, scaler: StandardScaler, except_keys: list[str] | None = None
    ) -> None:
        assert scaler.scale_ is not None, "Scaler not trained."
        except_keys = except_keys or []
        keys = [k for k in self.keys if k not in except_keys]
        keys.sort()
        for label in keys:
            df_data = {n: self.by_name(n).data[label] for n in self.names}
            df_data = pd.DataFrame(df_data).to_numpy()
            df_data = scaler.inverse_transform(df_data)

            for i, n in enumerate(self.names):
                self.by_name(n).data[label] = df_data[:, i]
                # self.by_name(n).data["y_1"] *= scaler.scale_[i]


class ModelIterator:
    """This class handles iterating over a dataset while applying a model.

    Args:
        onnx_path (str): The model to load (ONNX format).
        data (Dataset): The dataset to use (if given aas path, the dataset will be configured to
            match the provided ONNX model).
    """

    def __init__(self, onnx_path: str, data: Dataset) -> None:
        self._data = data
        self._len = len(data)

        x_shape, y_shape = self._data[0]
        self.x_shape, self.y_shape = x_shape.shape, y_shape.shape

        # create the ONNX session and find the input/output names
        self._session = onnxruntime.InferenceSession(onnx_path)
        self._i_onnx = [i.name for i in self._session.get_inputs()]
        self._o_onnx = [i.name for i in self._session.get_outputs()]

        logger.info("Found model input names: %s", self._i_onnx)
        logger.info("Found model output names: %s", self._o_onnx)

    def _init_container(self) -> dict[str, np.ndarray]:
        args = self._data.get_args()
        f_out = args.features_out
        sensor_output = {
            n: np.zeros(shape=(self._len, f_out), dtype=np.float32)
            for n in self._o_onnx
        }

        return sensor_output

    def run(self, batch_size: int = 64, num_workers: int = 4) -> IteratorResult:
        """This function executes a ModelCheckpoint on a given dataset. Then the model's output and input will
        be returned.

        Args:
            batch_size (int): The number of samples in a single pass. Defaults to 64.
            num_workers (int): Number of threads iterating the dataset. Defaults to 4.

        Returns:
            IteratorResult: The model's result on the given dataset.

        """
        sensor_output = self._init_container()

        limit = 0
        dataloader = DataLoader(
            self._data,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        for i, (x, y) in enumerate(tqdm(dataloader)):
            # run onnx inference
            inputs = [x.numpy().astype(np.float32), y.numpy().astype(np.float32)]
            inputs = {n: inputs[i] for i, n in enumerate(self._i_onnx)}
            onnx_output = list(self._session.run(None, inputs))

            if not isinstance(onnx_output, tuple):
                onnx_output = onnx_output

            if len(onnx_output[0].shape) == 3:
                onnx_output = [o[:, 0, :] for o in onnx_output]

            # extract the model output/input into the containers
            batch_size = len(sensor_output[self._o_onnx[0]][limit : limit + batch_size])
            for i, name in enumerate(self._o_onnx):
                sensor_output[name][limit : limit + batch_size] = onnx_output[i][
                    :batch_size
                ]

            limit += batch_size

        # ensure correct length
        sensor_output = {n: d[:limit] for n, d in sensor_output.items()}

        args = self._data.get_args()
        result = IteratorResult(
            prediction=sensor_output,
            names=args.names_out,
            keys=self._o_onnx,
            ts=self._data.get_timestamp(0),
        )

        return result
