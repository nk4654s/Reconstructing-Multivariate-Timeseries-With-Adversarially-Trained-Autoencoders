"""This module handles string to optimizer conversion."""
from typing import Type, Union

import torch.optim as optim

_optimizer_register = {
    "Adam": optim.Adam,
    "Adagrad": optim.Adagrad,
    "SparseAdam": optim.SparseAdam,
    "RMSprop": optim.RMSprop,
    "ASGD": optim.ASGD,
    "SGD": optim.SGD,
    "Adadelta": optim.Adadelta,
    "Adamax": optim.Adamax,
    "AdamW": optim.AdamW,
}


OPTIM_CLASS = Union[
        Type[optim.Adam],
        Type[optim.Adagrad],
        Type[optim.SparseAdam],
        Type[optim.RMSprop],
        Type[optim.ASGD],
        Type[optim.SGD],
        Type[optim.Adadelta],
        Type[optim.Adamax],
        Type[optim.AdamW]
    ]


def get_optimizer_by_name(name: str) -> OPTIM_CLASS:
    """Get optimizer by name.

    Args:
        name (str): Name of the optimizer to be retrieved

    Raises:
        ValueError: If the optimizer is unknown
        
    Returns:
        Optimizer: Optimizer
    """
    if name not in _optimizer_register:
        raise ValueError(f"{name} is not a valid optimizer")

    return _optimizer_register[name]
