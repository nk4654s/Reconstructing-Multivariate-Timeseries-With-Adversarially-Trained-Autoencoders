"""This module contains helper methods related to the models."""
import os
from typing import Dict
import torch.nn as nn
import yaml

from Comprisk.helper.file_structure import class_for_name


def init_sub_modules(path: str) -> Dict[str, nn.Module]:
    """This method loads the hyperparameters of an already trained model
    and initializes the model's sub_modules if existing.

    Args:
        patah (str): The folde rpath where the model is saved.
    """
    hparam_path = os.path.join(path, "hparams.yaml")
    with open(hparam_path, mode="r", encoding="UTF-8") as f:
        hparams = yaml.safe_load(f)
        sub_modules = {}
        for key, value in hparams["init_args"].items():
            if isinstance(value, dict):
                libpath = value["class_path"]
                module = class_for_name(*libpath.rsplit(".", maxsplit=1))
                sub_modules[key] = module(**value["init_args"])
        return sub_modules
