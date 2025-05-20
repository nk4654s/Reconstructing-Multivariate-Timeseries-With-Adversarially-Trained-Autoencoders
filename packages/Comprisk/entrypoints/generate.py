import os.path
from datetime import UTC, datetime

import pandas as pd
from Dataflux import logging_mode
from Dataflux.data import DataType, DataUpdater, EmptyScoreRegister, UpdateCallable
from Dataflux.logger import get_logger

from Comprisk.custom import DataModule
from Comprisk.entrypoints import EntrypointBase, EntrypointContext
from Comprisk.helper import IteratorResult, ModelCheckpoint, ModelIterator

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class CompriskUpdateCallable(UpdateCallable):
    """
    This callable sends the score values of the model to the updater class.

    Args:
        malfunction (pd.DataFrame): The dataframe containing the score values of
            the model.
    """

    def __init__(self, malfunction: IteratorResult) -> None:
        self._malfunction = malfunction.to_dataframe()[0]

    def get_update(self, start: int, end: int, total_key: str) -> pd.DataFrame:
        df = self._malfunction[self._malfunction.index >= start]
        df = df[df.index <= end]

        # calculate the mean as total machine score, other metrics might be better
        df[total_key] = df.min(axis=1)
        return df


class GeneratorEntrypoint(EntrypointBase):
    """
    This entrypoint loads a model and runs inference on the validation dataset. For
    convenience, the `Comprisk.entrypoints.FluxEntrypoint` should be called before this entrypoint in the pipeline.
    Then, by calling `comprisk-generate -c path_to_trained_model_config.yaml --data.validate path_to_config.yaml` the
    scores will be generated and uploaded to the same machine as stated in the config file. Note: by setting the
    `CAS_DEBUG_MODE` environment variable to the same UUID, the scores will be uploaded to a test device.
    """
    def __init__(self) -> None:
        super().__init__()

        self._updater_cls = DataUpdater
        self._check_point_cls = ModelCheckpoint

    def __call__(self, context: EntrypointContext, *args, **kwargs) -> EntrypointContext:
        # get the upload dataset which must be set as validation set, disable training parameters such as shuffle
        datamodule_args_modified = vars(context.config.data)

        # catch the case, where the machine is offline and the dataset is empty
        datamodule_args_modified["trainset"] = None
        data = DataModule(**datamodule_args_modified).valset
        logger.debug("Loaded (val) from %s", data.directory)
        if data is None:
            logger.warning("The dataset is empty, nothing to generate. Exiting gracefully.")
            return context

        # get the dataset's start and end time
        start = datetime.fromtimestamp(data.get_timestamp(0), tz=UTC)
        end = datetime.fromtimestamp(data.get_timestamp(-1), tz=UTC)

        # run inference on the validation set
        onnx_path = os.path.join(context.config_folder, "model.onnx")
        iterator = ModelIterator(onnx_path, data)
        result = iterator.run(batch_size=128, num_workers=2)

        # update the remote TSV files
        try:
            debug = os.environ.get("CAS_DEBUG_MODE", "0") == data.UUID
            updater = self._updater_cls(
                data.UUID,
                CompriskUpdateCallable(result),
                debug=debug,
            )
            updater.set_data_type(DataType.MEASUREMENTS)
            updater.update(start, end, data.get_args().skip)
        except EmptyScoreRegister as e:
            logger.error("Could not update files: %s", e)
            return context

        # remove the temporary downloaded dataset again
        # shutil.rmtree(get_dataset_path(args.dataset), ignore_errors = True)
        logger.info(
            "App finished from for machine %s (%s to %s)",
            start.strftime("%Y-%m-%d %H:%M:%S"),
            end.strftime("%Y-%m-%d %H:%M:%S"),
            data.UUID,
        )

        return context
