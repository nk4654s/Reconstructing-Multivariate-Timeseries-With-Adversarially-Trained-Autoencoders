from argparse import ArgumentParser
import os
import shutil
import subprocess
from typing import Tuple

import pandas as pd
import yaml

from Dataflux.pytorch import Dataset
from Dataflux.data import DataUpdater, UpdateCallable, DataType
from Comprisk.helper.model import ModelLoader


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


def get_dataset_args(model_folder: str) -> Tuple[int, int, int]:
    """
    This function gets the dataset arguments of the pipeline config given. Namely,
    the function locates and returns the `skip`, `in_sequence`, and `out_sequence`
    values.

    Args:
        model_folder (str): The path to the model root folder. Here, the config
            with which the model was trained must be located.
    
    Returns:
        skip (int): The number of samples to skip in the dataset, the length of the
            input sequence, and the length of the output sequence.
    """
    config_path = os.path.join(model_folder, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)["data"]
        skip = int(config.get("skip", 1))
        input_sequence = int(config.get("input_sequence", 1))
        output_sequence = int(config.get("output_sequence", 0))
        return (input_sequence, output_sequence, skip)


class CompriskUpdateCallable(UpdateCallable):
    """
    This callable sends the score values of the model to the updater class.

    Args:
        malfunction (pd.DataFrame): The dataframe containing the score values of
            the model.
        critical (pd.DataFrame): The dataframe containing the critical values
            of the model.
        total_key (str): The total malfunction probability key name.
    """
    def __init__(self, malfunction: pd.DataFrame, critical: pd.DataFrame) -> None:
        self._malfunction = malfunction
        self._critical = critical

    def get_update(self, start: int, end: int, total_key: str) -> pd.DataFrame:
        df = self._malfunction[self._malfunction.index  >= start]
        df = df[df.index <= end]

        # calculate the mean as total machine score, other metrics might be better
        df[total_key] = df.max(axis=1)
        return df


def cli_main():
    # define program arguments
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", type = str, required = True, help = "The model to load.")
    parser.add_argument("--dataset", "-d", type = str, required = True, help = "The dataset to use.")
    args = parser.parse_args()

    # download and process the dataset, the config file contains the requested
    # time frame to generate the scores for
    subprocess.run(["flux-pipeline", "-c", args.dataset])

    # load the trained model from disk and create the dataset object for the
    # previously downloaded dataset
    loader = ModelLoader(args.model)
    data = Dataset(get_dataset_path(args.dataset), *get_dataset_args(args.model))

    # execute the model on the dataset and calculate the scores
    score = loader.get_score_transformer()
    score_data = loader.predict_loop(data, step=score.step)
    score_data = score.transform(*score_data, static=False)

    # update the dataset's virtual sensors in the S3 buckets with the scores
    time_steps = (data.get_timestamp(0), data.get_timestamp(len(data) - 1))
    # FIXME: use `data.UUID`
    updater = DataUpdater("5ac7f03f-f4e6-4b15-9bc6-21db798f9d25", CompriskUpdateCallable(*score_data))
    updater.set_data_type(DataType.MEASUREMENTS)
    updater.update(*time_steps, data.get_meta().skip)

    # remove the temporary downloaded dataset again
    shutil.rmtree(get_dataset_path(args.dataset), ignore_errors = True)
    print(f"Generated and uploaded scores from {data.get_day(0)} to {data.get_day(len(data) - 1)}")


if __name__ == "__main__":
    cli_main()
