import os.path
import pandas as pd

from Dataflux.cli import FluxConfigManager
from Dataflux.pytorch import EmptyDatasetError
from Dataflux.data import DataUpdater, UpdateCallable, DataType
from Dataflux.pipeline import pipeline as flux_pipeline

from Comprisk.data import DataModule
from Comprisk.cli import CompriskLoadCLI
from Comprisk.helper import get_config_folder, ModelLoader
from Comprisk.model import BaseModel

from Comprisk import logger


class CompriskUpdateCallable(UpdateCallable):
    """
    This callable sends the score values of the model to the updater class.

    Args:
        malfunction (pd.DataFrame): The dataframe containing the score values of
            the model.
    """

    def __init__(self, malfunction: pd.DataFrame) -> None:
        self._malfunction = malfunction

    def get_update(self, start: int, end: int, total_key: str) -> pd.DataFrame:
        df = self._malfunction[self._malfunction.index >= start]
        df = df[df.index <= end]

        # calculate the mean as total machine score, other metrics might be better
        df[total_key] = df.max(axis=1)
        return df


def cli_main():
    cli = CompriskLoadCLI(BaseModel, DataModule, subclass_mode_model=True, run=False)

    base_path = get_config_folder(cli)
    logger.info("Logging model: %s", base_path)

    # download and process the dataset, the config file contains the requested
    # time frame to generate the scores for
    # If the dataset is given as a folder, there is no need to download it again
    datamodule_args_modified = cli.config.data.as_dict()
    datamodule_args_modified["shuffle"] = False
    if os.path.isfile(cli.config.dataset):
        flux = FluxConfigManager(cli.config.dataset)
        datamodule_args_modified["train"] = flux_pipeline(flux)
    else:
        datamodule_args_modified["train"] = cli.config.dataset

    # catch the case, where the machine is offline and the dataset is empty
    try:
        data = DataModule(**datamodule_args_modified).trainset
        if data is None:
            raise EmptyDatasetError()
    except EmptyDatasetError:
        logger.warning("The dataset is empty, nothing to generate. Exiting gracefully.")
        exit(0)

    # load the trained model from disk and create the dataset object for the
    # previously downloaded dataset
    loader = ModelLoader(base_path)

    # execute the model on the dataset and calculate the scores
    # TODO: load parameter `factor` and hand over to transform
    score = loader.get_score_transformer()
    logger.info("Loaded score transformer (mode=%s)", score.mode)
    score_data = loader.predict_loop(data)
    score_data = score.transform(*score_data, factors=0.1)

    # update the dataset's virtual sensors in the S3 buckets with the scores
    time_steps = (data.get_timestamp(0), data.get_timestamp(len(data) - 1))

    debug = os.environ.get("CAS_DEBUG_MODE", "0") == data.UUID
    updater = DataUpdater(data.UUID, CompriskUpdateCallable(score_data), debug=debug)
    updater.set_data_type(DataType.MEASUREMENTS)
    updater.update(*time_steps, data.get_args().skip)

    # remove the temporary downloaded dataset again
    # shutil.rmtree(get_dataset_path(args.dataset), ignore_errors = True)
    logger.info(
        "Finished uploading from %s to %s for machine %s",
        data.get_day(0),
        data.get_day(len(data) - 1),
        data.UUID,
    )


if __name__ == "__main__":
    cli_main()
