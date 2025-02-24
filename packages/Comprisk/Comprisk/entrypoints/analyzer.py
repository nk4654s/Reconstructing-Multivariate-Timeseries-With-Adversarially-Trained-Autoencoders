import os
import shutil

import numpy as np
import onnxruntime
from tqdm import tqdm
from uuid import UUID

import matplotlib.pyplot as plt

from Dataflux.pytorch import Dataset
from Dataflux.visualization import show_dataframe
from Dataflux.helper import get_sensors
from torch.utils.data import DataLoader
from Dataflux import logging_mode
from Dataflux.logger import get_logger

from Comprisk.anomaly import ScoreWrapper
from Comprisk.entrypoints import EntrypointBase, EntrypointContext
from Comprisk.helper import ModelCheckpoint, ModelIterator, Report


# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class AnalyzerEntrypoint(EntrypointBase):
    """
    This entrypoint takes a trained model and runs inference it on the validation set. Then the results are plotted in
    a graph located in the model's output folder. To customize the folder's name, use the `--prefix folder_name`
    argument.

    If a different dataset should be used, then overwrite the validation set using `--data.validate`.
    """
    def __init__(self) -> None:
        super().__init__()

        self._scores_path = None
        self._scores_summary_path = None
        self._full_frame_path = None

    def __call__(
        self, context: EntrypointContext, *args, **kwargs
    ) -> EntrypointContext:
        # load the dataset using the datamodule
        context.datamodule.set_path(None, context.config.data.validate)
        dataset = context.datamodule.valset

        # create the ONNX session and find the input/output names
        onnx_path = os.path.join(context.config_folder, "model.onnx")
        score_session = onnxruntime.InferenceSession(onnx_path)
        onnx_input_names = [i.name for i in score_session.get_inputs()]
        onnx_output_names = [i.name for i in score_session.get_outputs()]
        score_index = onnx_output_names.index("score")
        y_hat_index = onnx_output_names.index("y_hat")

        # run inference
        ts = []
        scores = []
        sensor_input = []
        sensor_output = []

        batch_size = 256
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
        for i, (x, y) in enumerate(tqdm(dataloader)):
            inputs = [np.array(x), np.array(y)]

            inputs = {n: inputs[i] for i, n in enumerate(onnx_input_names)}
            sensor_input += list(y[:, -1, :])

            onnx_output = list(score_session.run(None, inputs))
            scores += list(onnx_output[score_index][:, -1, :])
            sensor_output += list(onnx_output[y_hat_index][:, -1, :])

            ts += [dataset.get_timestamp(i * batch_size) + t for t in range(batch_size)]
        ts = ts[:len(sensor_output)]

        # create a report and generate the graphs
        name = "analysis" if not context.config.prefix else context.config.prefix
        report = Report(context.config_folder, name)
        report.generate_report_graphs(scores, sensor_input, sensor_output, ts)

        logger.info("Finished analyzing model on machine %s.", dataset.UUID)

        return context
