import os.path
import subprocess

import pandas as pd
import yaml

from Comprisk import logger
from Dataflux.pytorch import EmptyDatasetError
from Dataflux.data import DataUpdater, UpdateCallable, DataType

from Comprisk.data import DataModule
from Comprisk.cli import CompriskLoadCLI
from Comprisk.helper import get_config_folder, ModelLoader
from Comprisk.model import BaseModel


def get_dataset_path(dataset: str) -> str:
    """
    This function gets the dataset root folder path of the pipeline config given.

    Args:
        dataset (str): The path to the dataset yaml file.
    
    Args:
        dataset (str): The path to the dataset root folder.
    """
    with open(dataset, "r", encoding="utf-8") as f:
        return yaml.load(f, yaml.BaseLoader)["folder"]


class CompriskUpdateCallable(UpdateCallable):
    """
    This callable sends the score values of the model to the updater class.

    Args:
        malfunction (pd.DataFrame): The dataframe containing the score values of
            the model.
        total_key (str): The total malfunction probability key name.
    """
    def __init__(self, malfunction: pd.DataFrame) -> None:
        self._malfunction = malfunction

    def get_update(self, start: int, end: int, total_key: str) -> pd.DataFrame:
        df = self._malfunction[self._malfunction.index  >= start]
        df = df[df.index <= end]

        # calculate the mean as total machine score, other metrics might be better
        df[total_key] = df.max(axis=1)
        return df


def cli_main():
    logger.debug("Starting CompriskCLI")
    cli = CompriskLoadCLI(BaseModel, DataModule, subclass_mode_model=True, run=False)

    logger.debug("Loging model: %s", cli.model.__class__.__name__)
    base_path = get_config_folder(cli)

    # download and process the dataset, the config file contains the requested
    # time frame to generate the scores for
    # If the dataset is given as a folder, there is no need to download it again
    datamodule_args_modified = cli.config.data.as_dict()
    datamodule_args_modified["shuffle"] = False
    if os.path.isfile(cli.config.dataset):
        subprocess.run(["flux-pipeline", "-c", cli.config.dataset])
        datamodule_args_modified["train"] = get_dataset_path(cli.config.dataset)
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
    logger.debug("Loaded score transformer (mode=%s)", score.mode)
    score_data = loader.predict_loop(data)
    score_data = score.transform(*score_data, factors=0.1)

    # update the dataset's virtual sensors in the S3 buckets with the scores
    time_steps = (data.get_timestamp(0), data.get_timestamp(len(data) - 1))
    # FIXME: use `data.UUID`
    updater = DataUpdater("5ac7f03f-f4e6-4b15-9bc6-21db798f9d25", CompriskUpdateCallable(score_data))
    updater.set_data_type(DataType.MEASUREMENTS)
    updater.update(*time_steps, data.get_args().skip)

    # remove the temporary downloaded dataset again
    # shutil.rmtree(get_dataset_path(args.dataset), ignore_errors = True)
    print(f"Generated and uploaded scores from {data.get_day(0)} to {data.get_day(len(data) - 1)}")


if __name__ == "__main__":
    cli_main()
