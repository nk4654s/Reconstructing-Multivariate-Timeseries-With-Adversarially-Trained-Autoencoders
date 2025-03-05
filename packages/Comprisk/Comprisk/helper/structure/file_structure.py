"""This module contains helper methods to navigate the folder structure."""
import importlib
import os
from typing import List

import torch

from Dataflux import logging_mode
from Dataflux.logger import get_logger

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


def search_models(path: str, search_depth: int = 50) -> List[str]:
    """Searches ONNX models in a given folder recursively.

    Args:
        path (str): Where to search, subfolders will be scanned as well if `search_depth` > 1:
        search_depth (int): How deep to search recursively. Defaults to 50.

    Returns:
        List[str]: The list of found paths to ONNX models
    """
    result = []
    for sub_path in os.listdir(path):
        sub_path = os.path.join(path, sub_path)

        if os.path.isdir(sub_path):
            result += search_models(sub_path, search_depth=search_depth - 1)
            continue

        if sub_path.split(".")[-1] == "onnx":
            result.append(sub_path)

    return result


def load_model(path: str, eval_mode: bool = True):
    """This method loads the best checkpoint of a model. Requires checkpoints of model to have
    the filename structure `...val=<validation_loss>.ckpt`.

    Args:
        path (str): The path where the model checkpoints were saved.
        eval_mode (bool): Weather to set the loaded model to eval mode or not. Defaults to True.
    """
    # get the best checkpoint
    checkpoints = get_checkpoints(path)
    checkpoints = sorted(checkpoints, key=lambda x: float(x[:-5].split("val=")[-1]))
    if len(checkpoints) == 0:
        logger.warning("Could not find a checkpoint to load a model from. (%s)", path)
        return None

    ckpt_path = os.path.join(path, "checkpoints", checkpoints[0])

    # get the model configuration to find the model class
    model = torch.load(ckpt_path)

    # initialize the mode
    logger.debug("Loaded model class %s", model.__class__.__name__)
    logger.info("Loaded from from %s", checkpoints[0])
    if eval_mode:
        model.eval()

    return model
