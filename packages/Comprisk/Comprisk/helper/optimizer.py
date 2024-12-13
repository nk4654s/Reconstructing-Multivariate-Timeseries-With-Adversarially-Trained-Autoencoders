"""This module handles string to optimizer conversion."""
from typing import Type, Union
from torch.optim import *


_optimizer_register = {
    "Adam": Adam,
    "Adagrad": Adagrad,
    "SparseAdam": SparseAdam,
    "RMSprop": RMSprop,
    "ASGD": ASGD,
    "SGD": SGD,
    "Adadelta": Adadelta,
    "Adamax": Adamax,
    "AdamW": AdamW,
}


OPTIM_CLASS = Union[
        Type[Adam],
        Type[Adagrad],
        Type[SparseAdam],
        Type[RMSprop],
        Type[ASGD],
        Type[SGD],
        Type[Adadelta],
        Type[Adamax],
        Type[AdamW]
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
