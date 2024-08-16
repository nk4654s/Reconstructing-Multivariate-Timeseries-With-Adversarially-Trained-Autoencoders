"""This module loads a model using the **Comprisk.cli.CompriskCLI** and
applys the loaded model on the configured dataset. The prediction will be
safed next to the model's log folder as graph comparing prediction
vs. target."""

from typing import Tuple
import os
import torch
import pandas as pd
from tqdm import tqdm

# simple demo classes for your convenience
from Dataflux.pytorch import Metadata, Dataset
from Dataflux.visualization import show_dataframe
from Dataflux.helper import get_logger

from Comprisk.helper import get_config_folder, get_checkpoints, init_sub_modules
from Comprisk.model import BaseModel
from Comprisk.model.ae import AutoEncoder
from Comprisk.data import DataModule
from Comprisk.cli import CompriskCLI

logger = get_logger(__name__, path="./logs/logfile")

# FIXME: debug and check if the learning rate finder result is used (expected ~1h)
# TODO: run config with more than 1 dataloader fails in docker

def load_model(cli, eval_mode: bool = True) -> BaseModel:
    """This method loads the latest checkpoint of a model.

    Args:
        cli (CompriskCLI): The cli object created when starting an entrypoint.
        eval_mode (bool): Wether to set the loaded model to eval mode or not. Defaults to True.
    """
    base_path = get_config_folder(cli)
    checkpoints = get_checkpoints(cli)

    # 1. get dataset configuration: start, end, step size (size of input window)
    sub_modules = init_sub_modules(base_path)
    model = cli.model.__class__.load_from_checkpoint(
        os.path.join(base_path + "/checkpoints/", checkpoints[-1]),
        **sub_modules
    )
    if eval_mode:
        model.eval()
    return model


def get_tensor_from_dataset(dataset: Dataset, i: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """This method gets a sample from a dataset given the index and transforms it
    to a tensor including moving the tensor to the provided device.

    Args:
        dataset (Dataset): The dataset from which samples should be gathered.
        i (int): The index of the sample to get.
        device (str): The device to move the tensors to (e.g. 'cpu', 'cuda', or 'mps').
    
    Returns:
        Tuple[torch.tensor, torch.tensor]: Sensor and weather data as tensor.
    """
    sensors, weather, _ = dataset[i]

    sensors = torch.tensor(sensors, device=device)
    sensors = torch.unsqueeze(sensors, dim=0)
    weather = torch.tensor(weather, device=device)
    weather = torch.unsqueeze(weather, dim=0)
    return sensors, weather



def predict_dataset(dataset: Dataset, model: BaseModel) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """This method applies a model on a given dataset. For an AutoEncoder, the reconstructed
    output will be returned. For future prediction models, the prediction of the future is
    output. In addition, the original input data will be temporarily stored and returned.

    Args:
        dataset (Dataset): The dataset to use.
        model (BaseModel): The model to use.
    
    Returns
        Tuple[pd.DataFrame, pd.DataFrame]: Pandas dataframes containing the prediction and input.
    """
    meta: Metadata = dataset.get_meta()
    logger.debug("Predict on dataset, length=%s", len(dataset))
    logger.debug("Dataset input sequence: %s", meta.num_input_sequence)
    logger.debug("Dataset output sequence: %s", meta.num_output_sequence)

    # 1. iterate over dataset, with shuffle deactivated
    predicted = {key: [] for key in meta.column_names_sensor}
    inputs = {key: [] for key in meta.column_names_sensor}

    if isinstance(model, AutoEncoder):
        for i in tqdm(range(0, len(dataset), meta.num_input_sequence)):
            sensors, weather = get_tensor_from_dataset(dataset, i, model.device)
            target = model.predict({"sensors": sensors, "weather": weather}, True)[0]

            # store the model inputs/outputs to a temporary dict
            for j, key in enumerate(meta.column_names_sensor):
                predicted[key] += list(target[:, j])
                inputs[key] += list(sensors.cpu().numpy()[0, :, j])

    else:
        for i in tqdm(range(0, len(dataset), meta.num_output_sequence)):
            sensors, weather = get_tensor_from_dataset(dataset, i, model.device)
            target = model.predict({"sensors": sensors, "weather": weather}, True)[0]

            # store the model inputs/outputs to a temporary dict
            for j, key in enumerate(meta.column_names_sensor):
                if i == 0:
                    predicted[key] += list(sensors.cpu().numpy()[0, :, j])
                if i % meta.num_input_sequence == 0:
                    inputs[key] += list(sensors.cpu().numpy()[0, :, j])
                predicted[key] += list(target[:, j])

    # 4. finalize dataframe, give columns names matching input dataframe column names
    logger.debug("Creating 'prediction' Dataframe of gathered data")
    predicted = pd.DataFrame(predicted)

    logger.debug("Creating 'inputs' Dataframe of gathered data")
    inputs = pd.DataFrame(inputs)
    return predicted, inputs


def cli_main():
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    logger.debug("Starting CompriskCLI")
    cli = CompriskCLI(BaseModel, DataModule, subclass_mode_model=True, run=False)

    logger.debug("Loging model: %s", cli.model.__class__.__name__)
    model = load_model(cli, eval_mode=True)
    #model.set_monte_carlo(20)

    dataset = cli.datamodule.dataset
    predicted, inputs = predict_dataset(dataset, model)

    # visualize
    sensors = [
        "SaugtemperaturZweiteStufeTETIT74341 (°C)",
        "DifferenzdruckTriebwerksölPDIT74553 (bar)",
        "KühlwassertemperaturnachKühlerTETIT74422 (°C)",
        "ÖltemperaturVorlaufTriebwerkTETIT74538 (°C)",
        "Motorlager (°C)",
        "SaugdruckErsteStufePIT74319 (bar)",
        "SchwingungsmessungTriebwerkVT74280 (mm/s)",
        "KompressordrehzahlSE74201 (rpm)",
        "EndtemperaturErsteStufeTETIT74334 (°C)",
        "EndtemperaturZweiteStufeTETIT74371 (°C)",
    ]

    logger.debug("Visualizing whole dataset (samples=%s)", len(dataset))
    base_path = get_config_folder(cli)
    show_dataframe(
        sensors,
        [predicted, inputs],
        ["Prediction", "Target"],
        grid_size=5,
        save=os.path.join(base_path, f"prediction-{len(dataset)}.png"),
    )

    meta: Metadata = dataset.get_meta()
    three_days = (24 * 60 * 60) // meta.skip * 3
    logger.debug("Visualizing reduced dataset (samples=%s)", 3 * three_days)
    show_dataframe(
        sensors,
        [predicted[:three_days], inputs[:three_days]],
        ["Prediction", "Target"],
        grid_size=5,
        save=os.path.join(base_path, f"prediction-{three_days}.png"),
    )


if __name__ == "__main__":
    cli_main()
