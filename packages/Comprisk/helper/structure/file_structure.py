"""This module contains helper methods to navigate the folder structure."""

import importlib
import os
from copy import deepcopy

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


def get_config_folder(cli) -> str:
    """Extracts the config folder from the command line arguments.

    Args:
        cli (CLI): The command line interface object. To use this method,
            cli must be called with the argument `-c`. Otherwise, no path
            will be available.

    Returns:
        str: The config folder path from the command line arguments.
    """
    config_path = cli.parser.args[cli.parser.args.index("-c") + 1]
    base_path = os.path.dirname(config_path)
    return base_path


def deepvars(val) -> dict:
    val = deepcopy(val)
    res = vars(val)
    for key in res:
        if hasattr(res[key], "__dict__"):
            res[key] = deepvars(res[key])
    return res
