import torch
from torch import nn, Tensor


class ScatteredGridInterpolator(nn.Module):
    """The ScatteredGridInterpolated gets a grid of support points and enables sampling points between cells. To
    achieve this, the two nearest neighbours were searched and a linear interpolation of the output is applied.
    Neighbours are defined by their euclidean distance to the input.

    > Note: This module disables the gradient to prevent changing the grid.

    Args:
        gx (Tensor): The input coordinated.
        gy (Tensor): The output vectors according to the input coordinates.
    """
    def __init__(self, gx, gy):
        super().__init__()

        self._gx = nn.Parameter(gx)
        self._gy = nn.Parameter(gy)

    @torch.no_grad()
    def forward(self, x: Tensor):
        # Compute squared Euclidean distance between x and all gx points
        dists = torch.sqrt(torch.sum((x - self._gx) ** 2, dim=-1))

        # Get indices of the two nearest neighbors
        knn_idx = torch.topk(dists, k=2, largest=False, dim=1).indices  # Shape: (batch, 2)

        # Get the two nearest neighbor points and outputs
        gx, gy = self._gx.unsqueeze(1), self._gy.unsqueeze(1)
        x0, x1 = gx[knn_idx[:, 0], :], gx[knn_idx[:, 1], :]  # (batch, samples, num_inputs)
        y0, y1 = gy[knn_idx[:, 0], :], gy[knn_idx[:, 1], :]  # (batch, samples, num_outputs)

        # Compute interpolation weights (based on inverse distance)
        d0, d1 = torch.norm(x - x0, dim=-1, keepdim=True), torch.norm(x - x1, dim=-1, keepdim=True)
        weight = d1 / (d0 + d1)  # Linear interpolation weight
        weight = torch.nan_to_num(weight, nan=1.0)

        # Perform interpolation
        y_interp = y0 * weight + y1 * (1 - weight)  # Shape: (batch, num_outputs)

        return y_interp
