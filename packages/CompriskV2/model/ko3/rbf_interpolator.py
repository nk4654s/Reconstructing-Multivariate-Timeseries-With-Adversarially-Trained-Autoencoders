import torch

def gaussian_kernel(x, y, **kwargs):
    """
    Gaussian RBF kernel function.
    Args:
        x: Input tensor of shape (N, D)
        y: Center points tensor of shape (M, D)
    Returns:
        Kernel matrix of shape (N, M)
    """
    sigma = kwargs.get("sigma", 1.0)
    # Handle batched or unbatched input
    if x.dim() == 2 and y.dim() == 2:
        x_sq = torch.sum(x ** 2, dim=1).unsqueeze(1)  # (N, 1)
        y_sq = torch.sum(y ** 2, dim=1).unsqueeze(0)  # (1, M)
        dist_sq = x_sq + y_sq - 2 * torch.matmul(x, y.T)
    else:
        # Batched version
        # x: (B, N, D), y: (B, M, D)
        x_sq = (x ** 2).sum(dim=2, keepdim=True)  # (B, N, 1)
        y_sq = (y ** 2).sum(dim=2).unsqueeze(1)   # (B, 1, M)
        dist_sq = x_sq + y_sq - 2 * torch.matmul(x, y.transpose(1, 2))  # (B, N, M)

    return torch.exp(-0.5 * dist_sq / (sigma ** 2))


def linear_kernel(x, y, **kwargs):
    """
    Linear kernel function for RBF interpolation.
    Args:
        x: Input tensor of shape (N, D)
        y: Center points tensor of shape (M, D)
    Returns:
        Kernel matrix of shape (N, M)
    """
    if x.dim() == 2 and y.dim() == 2:
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (N, M, D)
        r = torch.norm(diff, dim=2)
    else:
        # x, y: (B, N, D), (B, M, D)
        diff = x.unsqueeze(2) - y.unsqueeze(1)  # (B, N, M, D)
        r = torch.norm(diff, dim=3)  # (B, N, M)
    return -r


kernels = {
    "linear": linear_kernel,
    "gaussian": gaussian_kernel
}


class RBFGridInterpolator(torch.nn.Module):
    def __init__(self, x, y, kernel="linear", **kwargs):
        super().__init__()
        assert kernel in kernels, f"given kernel '{kernel}' does not exist."
        assert len(x) == len(y), f"'x' and 'y' must have the same length, got {len(x)} and {len(y)}"
        assert len(x) > 1, f"More samples required, got {len(x)} but at least 2 are needed."

        self.register_buffer("x", x)
        self.register_buffer("y", y)

        self.kernel_args = kwargs
        self.kernel = kernels[kernel]

        phi_xx = self.kernel(x, x, **kwargs)  # (N, N)

        # Add regularization
        eps = kwargs.get("epsilon", 1e-6)
        phi_xx = phi_xx + eps * torch.eye(phi_xx.size(0), device=x.device, dtype=x.dtype)

        self.register_buffer("weights", torch.linalg.solve(phi_xx, y))

    @torch.no_grad()
    def forward(self, x_new):
        if x_new.dim() == 2:
            # (N, D) → unbatched
            phi = self.kernel(x_new, self.x, **self.kernel_args)  # (N, centers)
            return phi @ self.weights
        elif x_new.dim() == 3:
            # (B, N, D)
            # Expand self.x to match batch: (B, M, D)
            B = x_new.size(0)
            x_centers = self.x.unsqueeze(0).expand(B, -1, -1)  # (B, M, D)
            phi = self.kernel(x_new, x_centers, **self.kernel_args)  # (B, N, M)
            return torch.matmul(phi, self.weights)  # (B, N, output_dim)
        else:
            raise ValueError("Input x_new must be of shape (N, D) or (B, N, D)")
