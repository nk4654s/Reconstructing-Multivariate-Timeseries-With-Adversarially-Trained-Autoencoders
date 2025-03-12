"""This module contains various loss functions."""
import torch
from torch import nn

_loss_register = {
    "MSELoss": nn.MSELoss(),
    "RMSELoss": lambda y1, y2: torch.sqrt(nn.MSELoss()(y1, y2)),
    "CELoss": nn.CrossEntropyLoss(),
    "L1Loss": nn.L1Loss(),
    "BCELoss": nn.BCELoss(),
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
    "KLDivLoss": nn.KLDivLoss(),
    "GaussianNLLLoss": nn.GaussianNLLLoss()
}


def r2_score(i, t):
    mean = t.mean(axis=0)
    ss_regression = ((i - mean) ** 2).sum(axis=0)
    ss_total = ((t - mean) ** 2).sum(axis=0)
    r2_score = ss_regression / ss_total

    return (1 / r2_score.clip(1, None)).mean()


def get_loss_by_name(name: str) -> nn.Module:
    """Get loss function by name.

    Args:
        name (str): Name of the loss function to be retrieved.

    Raises:
        ValueError: If the loss is unknown.
        
    Returns:
        nn.Module: Loss function.
    """
    if name not in _loss_register:
        raise ValueError(f"Unknown loss {name}")

    return _loss_register[name]
