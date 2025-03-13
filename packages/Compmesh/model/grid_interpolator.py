import torch
from torch import nn, Tensor


class ScatteredGridInterpolator(nn.Module):
    def __init__(self, gx: torch.Tensor, gy: torch.Tensor):
        super().__init__()

        self._gx = gx
        self._gy = gy

    @torch.no_grad()
    def forward(self, x: Tensor):
        # Compute squared Euclidean distance between x and all gx points
        dists = torch.cdist(x, self._gx)  # Shape: (batch, samples)

        # Get indices of the two nearest neighbors
        knn_idx = torch.topk(dists, k=2, largest=False).indices  # Shape: (batch, 2)

        # Get the two nearest neighbor points and outputs
        x0, x1 = self._gx[knn_idx[:, :, 0]], self._gx[knn_idx[:, :, 1]]  # (batch, samples, num_inputs)
        y0, y1 = self._gy[knn_idx[:, :, 0]], self._gy[knn_idx[:, :, 1]]  # (batch, samples, num_outputs)

        # Compute interpolation weights (based on inverse distance)
        d0, d1 = torch.norm(x - x0, dim=1, keepdim=True), torch.norm(x - x1, dim=1, keepdim=True)
        weight = d1 / (d0 + d1)  # Linear interpolation weight

        # Perform interpolation
        y_interp = y0 * weight + y1 * (1 - weight)  # Shape: (batch, num_outputs)

        return y_interp
