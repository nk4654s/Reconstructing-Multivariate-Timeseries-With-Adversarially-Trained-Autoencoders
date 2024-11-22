"""This module loads a model using the **Comprisk.cli.CompriskCLI** and
applys the loaded model on the configured dataset. The prediction will be
safed next to the model's log folder as graph comparing prediction
vs. target."""
import os
import shutil

from Dataflux.pytorch import Metadata, Dataset
from Dataflux.visualization import show_dataframe
from Dataflux.helper import get_logger

from Comprisk.helper import get_config_folder, ModelLoader
from Comprisk.model import BaseModel
from Comprisk.data import DataModule
from Comprisk.cli import CompriskLoadCLI

logger = get_logger(__name__, path="./logs/logfile")


def cli_main():
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    logger.debug("Starting CompriskCLI")
    cli = CompriskLoadCLI(BaseModel, DataModule, subclass_mode_model=True, run=False)

    logger.debug("Loging model: %s", cli.model.__class__.__name__)
    base_path = get_config_folder(cli)

    # load the dataset and configure it to work with the trained model
    dataset = cli.datamodule.trainset
    if cli.config.dataset:
        meta = dataset.get_meta()
        dataset = Dataset(cli.config.dataset,
                          meta.num_input_sequence,
                          meta.num_output_sequence,
                          meta.skip)

    # load the model with the helper class, then apply the model on the dataset
    loader = ModelLoader(base_path)
    score = loader.get_score_transformer()
    target, predicted = loader.predict_loop(dataset, modifier=(
        cli.config.anomaly.name,
        cli.config.anomaly.factor
    ), step=cli.config.step or score.step)

    # visualize the loaded dataset
    prefix = "prediction"
    base_path = os.path.join(get_config_folder(cli), prefix)

    # create prefix sub-folders
    if cli.config.prefix:
        prefix = cli.config.prefix
        base_path = os.path.join(get_config_folder(cli), prefix)

    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path)

    # save the whole prediction of the dataset
    meta: Metadata = dataset.get_meta()
    days = round(len(dataset) * meta.skip // 60 // 60 // 24)
    logger.debug("Visualizing whole dataset (samples=%s)", len(dataset))
    show_dataframe(
        [predicted, target],
        ["Prediction", "Target"],
        grid_size=5,
        save=os.path.join(base_path, f"days={days}.png"),
    )

    # save the prediction of three days of the dataset
    twelve_hrs = (12 * 60 * 60) // meta.skip
    logger.debug("Visualizing reduced dataset (samples=%s)", 3 * twelve_hrs)
    show_dataframe(
        [predicted[:twelve_hrs], target[:twelve_hrs]],
        ["Prediction", "Target"],
        grid_size=5,
        save=os.path.join(base_path, "hrs=12.png"),
    )

    # transformes the loss between prediction and target to a score between 0 and 1.
    score_vis_path = os.path.join(base_path, "score-transformed.png")
    score.transform(predicted, target, visualize=score_vis_path, static=cli.config.static_scores)


if __name__ == "__main__":
    cli_main()
