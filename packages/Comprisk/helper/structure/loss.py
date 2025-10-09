"""This module contains various loss functions."""
import torch
from torch import nn

class CustomBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, weight: float = 1.0):
        super(CustomBCEWithLogitsLoss, self).__init__()
        self._weight = weight

    def forward(self, real: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        real = torch.sigmoid(real)
        eps = 1e-7
        real = torch.clamp(real, eps, 1 - eps) # prevent sigmoid ouput of exactly 1 or 0
        loss = - self._weight * (target * torch.log(real) - (1-target) * torch.log(1 - real))
        return  loss.mean()

_loss_register = {
    "MSELoss": nn.MSELoss(),
    "RMSELoss": lambda y1, y2: torch.sqrt(nn.MSELoss()(y1, y2)),
    "CELoss": nn.CrossEntropyLoss(),
    "L1Loss": nn.L1Loss(),
    "BCELoss": nn.BCELoss(),
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
    "CustomBCEWithLogitsLoss": CustomBCEWithLogitsLoss(),
    "KLDivLoss": nn.KLDivLoss(),
    "GaussianNLLLoss": nn.GaussianNLLLoss()
}


def r2_score(i, t):
    mean = t.mean(axis=0)
    ss_regression = ((i - mean) ** 2).sum(axis=0)
    ss_total = ((t - mean) ** 2).sum(axis=0)
    r2_score = ss_regression / ss_total

    return (1 / r2_score.clip(1, None)).mean()

def batch_wasserstein(x_hat, x):
    """
    Vectorized Wasserstein distance over batch and features.
    x, x_hat: [batch, length, features]

    approximation of Wasserstein distance that can be computed on GPU as to not bottleneck training loop

    CHATGPT FUNCTION !!!!!!!!!!!!!!!!!!!! REVISE IF NEEDED
    """
    # Sort along the sequence dimension
    x_sorted, _ = torch.sort(x, dim=1)
    x_hat_sorted, _ = torch.sort(x_hat, dim=1)

    # Mean absolute difference across sequence
    distances = torch.mean(torch.abs(x_sorted - x_hat_sorted), dim=1)  # [batch, features]

    # Average over batch + features
    return distances.mean()



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
