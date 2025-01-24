"""This module contains helper methods to navigate the folder structure."""
import importlib
import os
from typing import List, Union, Tuple, Type, Dict

import yaml
from torch import nn

from Dataflux import logging_mode
from Dataflux.logger import get_logger

from Comprisk.model import BaseModel

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)

def class_for_name(module_name: str, class_name: str):
    """This function finds the class given the module and class name
    
    Args:
        module_name (str): The module path where the class is located.
        class_name (str): The name of the class.
    """
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def get_config_path(cli) -> str:
    """Extracts the config path from the command line arguments.

    Args:
        cli (CLI): The command line interface object. To use this method,
            cli must be called with the argument `-c`. Otherwise, no path
            will be available.

    Returns:
        str: The config path from the command line arguments.
    """
    config_path = cli.parser.args[cli.parser.args.index("-c") + 1]
    return config_path


def get_config_folder(cli) -> str:
    """Extracts the config folder from the command line arguments.

    Args:
        cli (CLI): The command line interface object. To use this method,
            cli must be called with the argument `-c`. Otherwise, no path
            will be available.

    Returns:
        str: The config folder path from the command line arguments.
    """
    config_path = get_config_path(cli)
    base_path = os.path.dirname(config_path)
    return base_path


def get_checkpoints(path: str) -> List[str]:
    """Finds the checkpoints in the config folder from the command line
    arguments.

    Args:
        path (str): The path to a model's output folder. This folder must contain
            `checkpoints/`

    Returns:
        List[str]: The list of checkpoint paths, sorted.
    """
    base_path = os.path.join(path, "checkpoints")
    if not os.path.exists(base_path):
        return []

    checkpoints = sorted([_ for _ in os.listdir(base_path) if "ckpt" in _])
    return checkpoints


def init_modules(path: str) -> Tuple[Type, Dict[str, nn.Module]]:
    """This method loads the hyperparameters of an already trained model
    and initializes the model's sub_modules if existing.

    Args:
        path (str): The folder path where the model is saved.

    Returns:
        Tuple: The class of the model to load and a dictionary of
            initialized submodules modules.
    """
    hparam_path = os.path.join(path, "hparams.yaml")
    with open(hparam_path, mode="r", encoding="UTF-8") as f:
        hparams = yaml.safe_load(f)
        sub_modules = {}

        for drop in [key for key in hparams.keys() if key.startswith("_")]:
            if drop == "class_path":
                continue
            del hparams[drop]

        for key, value in hparams.items():
            if drop == "class_path":
                continue

            if isinstance(value, dict):
                libpath = value["class_path"]
                module = class_for_name(*libpath.rsplit(".", maxsplit=1))
                sub_modules[key] = module(**value["init_args"])
            else:
                sub_modules[key] = value

        model_class = class_for_name(*hparams["class_path"].rsplit(".", maxsplit=1))
        return model_class, sub_modules


def load_model(path: str, eval_mode: bool = True) -> Union[BaseModel, None]:
    """This method loads the best checkpoint of a model. Requires checkpoints of model to have
    the filename structure `...val=<validation_loss>.ckpt`.

    Args:
        path (str): The path where the model checkpoints were saved.
        eval_mode (bool): Weather to set the loaded model to eval mode or not. Defaults to True.
    """
    checkpoints = get_checkpoints(path)
    checkpoints = sorted(checkpoints, key=lambda x: float(x[:-5].split("val=")[-1]))
    if len(checkpoints) == 0:
        logger.warning("Could not find a checkpoint to load a model from. (%s)", path)
        return None

    # 1. get dataset configuration: start, end, step size (size of input window)
    model_class, sub_modules = init_modules(path)
    model = model_class.load_from_checkpoint(
        os.path.join(path, "checkpoints", checkpoints[0]), **sub_modules
    )

    logger.info("Loaded model %s from %s", checkpoints[0], path)
    if eval_mode:
        model.eval()
    return model
