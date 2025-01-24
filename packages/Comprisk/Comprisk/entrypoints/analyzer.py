"""This module loads a model using the **Comprisk.cli.CompriskCLI** and
apply the loaded model on the configured dataset. The prediction will be
saved next to the model's log folder as graph comparing prediction
vs. target."""

import os
import shutil
from uuid import UUID

from Dataflux.pytorch import Dataset, DatasetArgs
from Dataflux.visualization import show_dataframe
from Dataflux.helper import get_sensors
from Dataflux import logging_mode
from Dataflux.logger import get_logger

from Comprisk.helper.structure import get_config_folder
from Comprisk.helper import ModelLoader
from Comprisk.model import BaseModel
from Comprisk.data import DataModule
from Comprisk.cli import CompriskLoadCLI


# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


def cli_main():
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    logger.debug("Starting CompriskCLI")
    cli = CompriskLoadCLI(BaseModel, DataModule, subclass_mode_model=True, run=False)

    base_path = get_config_folder(cli)
    logger.info("Logging model: %s", base_path)

    # load the dataset and configure it to work with the trained model
    cli.datamodule.set_path(cli.config.dataset, None)
    dataset = cli.datamodule.trainset
    if cli.config.dataset:
        args = dataset.get_args().as_dict()
        args["path"] = cli.config.dataset
        dataset = Dataset(**args)

    # load the model with the helper class, then apply the model on the dataset
    loader = ModelLoader(base_path)
    score = loader.get_score_transformer()
    logger.info("Loaded score transformer (mode=%s)", score.mode)

    logger.info(
        "Run prediction loop on dataset (samples=%s) which can take some time.",
        len(dataset),
    )
    target, predicted = loader.predict_loop(dataset)

    # visualize the loaded dataset
    prefix = "analysis"
    base_path = os.path.join(get_config_folder(cli), prefix)

    # create prefix sub-folders
    if cli.config.prefix:
        prefix = cli.config.prefix
        base_path = os.path.join(get_config_folder(cli), prefix)

    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path)

    # transforms the loss between prediction and target to a score between 0 and 1.
    score_vis_path = os.path.join(base_path, "score-transformed.png")
    # TODO get the 'factors' parameter from a config file which should be generated where the model got stored
    score.transform(predicted, target, visualize=score_vis_path, factors=0.1)

    # rename the columns, which might be required
    logger.info(
        "Downloading uuid to sensor table for sensor renaming in visualizations."
    )
    sensors_name_dict = get_sensors(UUID(dataset.UUID))
    predicted = predicted.rename(columns=sensors_name_dict, errors="ignore")
    target = target.rename(columns=sensors_name_dict, errors="ignore")

    # save the whole prediction of the dataset
    dataset_args: DatasetArgs = dataset.get_args()
    days = round(len(dataset) * dataset_args.skip // 60 // 60 // 24)
    logger.debug("Visualizing whole dataset (samples=%s)", len(dataset))
    show_dataframe(
        [predicted, target],
        ["Prediction", "Target"],
        grid_size=5,
        save=os.path.join(base_path, f"days={days}.png"),
    )

    # save the prediction of three days of the dataset
    twelve_hrs = (12 * 60 * 60) // dataset_args.skip
    logger.debug("Visualizing reduced dataset (samples=%s)", 3 * twelve_hrs)
    show_dataframe(
        [predicted.iloc[:twelve_hrs], target.iloc[:twelve_hrs]],
        ["Prediction", "Target"],
        grid_size=5,
        save=os.path.join(base_path, "hrs=12.png"),
    )

    logger.info(
        "Finished analyzing model %s on machine %s.",
        loader.model.__class__.__name__,
        dataset.UUID,
    )


if __name__ == "__main__":
    cli_main()
