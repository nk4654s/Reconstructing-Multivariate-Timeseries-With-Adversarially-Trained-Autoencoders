"""This module contains helper methods to navigate the folder structure."""
import os
from typing import List


def get_config_path(cli) -> str:
    """Extracts the config path from the command line arguments.

    Args:
        cli (CLI): The command line interface object. To use this method,
            cli must be called with the argument `-c`. Otherwise, no path
            will be avialable.

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
            will be avialable.

    Returns:
        str: The config folder path from the command line arguments.
    """
    config_path = get_config_path(cli)
    base_path = os.path.dirname(config_path)
    return base_path


def get_checkpoints(cli) -> List[str]:
    """Finds the checkpoints in the config folder from the command line
    arguments.

    Args:
        cli (CLI): The command line interface object. To use this method,
            cli must be called with the argument `-c`. Otherwise, no path
            will be avialable.

    Returns:
        List[str]: The list of checkpoint paths, sorted.
    """
    base_path = get_config_folder(cli)
    checkpoints = sorted([_ for _ in os.listdir(base_path) if "ckpt" in _])
    return checkpoints
