"""This module loads a model using the **Comprisk.cli.CompriskCLI** and
applys the loaded model on the configured dataset. The prediction will be
safed next to the model's log folder as graph comparing prediction
vs. target."""

from typing import Dict
import importlib
import os
import torch
import torch.nn as nn
import pandas as pd
import yaml
from tqdm import tqdm

# simple demo classes for your convenience
from Dataflux.pytorch import Metadata
from Dataflux.visualization import show_dataframe
from Dataflux.helper import get_logger

from Comprisk.helper import get_config_folder, get_checkpoints, init_sub_modules
from Comprisk.model import BaseModel
from Comprisk.data import DataModule
from Comprisk.cli import CompriskCLI

logger = get_logger(__name__, path="./logs/logfile")

# FIXME: debug and check if the learning rate finder result is used (expected ~1h)
# TODO: create cuda enabled conda environment (expected ~1h)
# TODO: Setup Ubuntu machine for server training with CUDA and docker (expected <1h)
# TODO: run config with more than 1 dataloader fails in docker


def cli_main():
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    logger.debug("Starting CompriskCLI")
    cli = CompriskCLI(BaseModel, DataModule, subclass_mode_model=True, run=False)

    base_path = get_config_folder(cli)
    checkpoints = get_checkpoints(cli)

    # 1. get dataset configuration: start, end, step size (size of input window)
    sub_modules = init_sub_modules(base_path)
    model = cli.model.__class__.load_from_checkpoint(
        os.path.join(base_path, checkpoints[-1]),
        **sub_modules
    )
    dataset = cli.datamodule.dataset
    meta: Metadata = dataset.get_meta()

    # 2. iterate over dataset, with shuffle deactivated
    predicted = {key: [] for key in meta.column_names_sensor}
    inputs = {key: [] for key in meta.column_names_sensor}
    for i in tqdm(range(0, len(dataset) // 10, meta.num_input_sequence)):
        sensors, weather, _ = dataset[i]

        sensors = torch.tensor(sensors, device=model.device)
        sensors = torch.unsqueeze(sensors, dim=0)
        weather = torch.tensor(weather, device=model.device)
        weather = torch.unsqueeze(weather, dim=0)

        # 3. pass sample into model, store prediction in list/numpy/dataframe
        target = model.predict({"sensors": sensors, "weather": weather}, as_array=True)[
            0
        ]

        for i, key in enumerate(meta.column_names_sensor):
            predicted[key] += list(target[:, i])
            inputs[key] += list(sensors.cpu().numpy()[0, :, i])

    # 4. finalize dataframe, give columns names matching input dataframe column names
    predicted = pd.DataFrame(predicted)
    inputs = pd.DataFrame(inputs)

    # 5. visualize
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
    show_dataframe(
        sensors,
        [predicted, inputs],
        ["Prediction", "Target"],
        grid_size=5,
        save=os.path.join(base_path, "prediction.png"),
    )


if __name__ == "__main__":
    cli_main()
