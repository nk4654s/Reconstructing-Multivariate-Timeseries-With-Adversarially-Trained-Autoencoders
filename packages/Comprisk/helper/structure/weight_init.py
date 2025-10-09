"""This module handles string to weight initialization conversion."""
from typing import Type

import torch.nn.init as init

_weight_init_register = {
    "He_normal": init.kaiming_normal_,
    "He_uniform": init.kaiming_uniform_,
    "Xavier_normal": init.xavier_normal_,
    "Xavier_uniform": init.xavier_uniform_,
    "Orthogonal": init.orthogonal_
}

WEIGHT_INIT_CLASS = Type[init.kaiming_normal] | Type[init.kaiming_uniform] | Type[init.xavier_normal] | \
              Type[init.xavier_uniform] | Type[init.orthogonal]


def get_weight_init_by_name(name: str) -> WEIGHT_INIT_CLASS:
    """Get weight initialization by name.

    Args:
        name (str): Name of the weight initialization to be retrieved

    Raises:
        ValueError: If the weight initialization is unknown

    Returns:
        TODO
    """
    if name not in _weight_init_register:
        raise ValueError(f"{name} is not a valid weight initialization")

    return _weight_init_register[name]