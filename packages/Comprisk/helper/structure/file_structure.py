"""This module contains helper methods to navigate the folder structure."""

import importlib
import os
import shutil
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
    if os.path.basename(path) == "derived_from":
        return []

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


def copy_folder(
    source: str, destination: str, exclude_folders: list[str] = None, exclude_files: list[str] = None
) -> None:
    """
    Copies a folder from the source to the destination, excluding specified subfolders.

    Args:
        source (str): The path to the folder that needs to be copied.
        destination (str): The path where the folder should be copied.
        exclude_folders (list of str): A list of subfolder names to exclude from the copy process.
        exclude_files (list of str): A list of file names (can be substrings) to exclude from the copy process.

    Returns:
        None
    """
    exclude_folders = exclude_folders or []
    exclude_files = exclude_files or []

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination):
        os.makedirs(destination)

    for root, dirs, files in os.walk(source):
        # Filter out the directories to exclude
        dirs[:] = [d for d in dirs if d not in exclude_folders]

        # Construct the destination path for the current root
        relative_path = os.path.relpath(root, source)
        destination_path = os.path.join(destination, relative_path)

        # Create the destination directory if it doesn't exist
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        # Copy the files to the destination
        for f in files:
            skip = sum([1 for _ in exclude_files if _ in f])
            if skip:
                continue
            source_file_path = os.path.join(root, f)
            destination_file_path = os.path.join(destination_path, f)
            shutil.copy2(source_file_path, destination_file_path)
