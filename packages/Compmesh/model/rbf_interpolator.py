import torch
from torch import nn, Tensor
from torchrbf import RBFInterpolator


class RBFGridInterpolator(nn.Module):
    def __init__(self, gx: Tensor, gy: Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._out = gy.size(-1)
        self._rbf = RBFInterpolator(gx, gy, kernel="quintic")

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        output = torch.empty(size=(x.size(0), x.size(1), self._out), device=x.device)
        for i in range(x.size(1)):
            output[:, i, :] = self._rbf(x[:, i, :])

        return output