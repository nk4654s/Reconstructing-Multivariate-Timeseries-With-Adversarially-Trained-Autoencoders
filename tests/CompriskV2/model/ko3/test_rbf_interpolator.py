import torch
from CompriskV2.model.ko3 import RBFGridInterpolator


def test_gaussian_kernel():
    x = torch.Tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    y = torch.Tensor([[0.0], [1.0], [2.0]])
    model = RBFGridInterpolator(x, y, kernel="gaussian")

    y1 = model(torch.tensor([[0.5, 1.0]]))
    assert torch.allclose(y1, torch.Tensor([[0.4232]]), rtol=1e-3)

    y1 = model(torch.tensor([[1.5, 3.0]]))
    assert torch.allclose(y1, torch.Tensor([[1.4865]]), rtol=1e-3)

    y1 = model(torch.tensor([[3.0, 6.0]]))
    assert torch.allclose(y1, torch.Tensor([[0.1585]]), rtol=1e-3)


def test_linear_kernel():
    x = torch.Tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    y = torch.Tensor([[0.0], [1.0], [2.0]])
    model = RBFGridInterpolator(x, y, kernel="linear")

    y1 = model(torch.Tensor([[0.5, 1.0]]))
    assert torch.allclose(y1, torch.Tensor([[0.5]]))

    y1 = model(torch.Tensor([[1.5, 3.0]]))
    assert torch.allclose(y1, torch.Tensor([[1.5]]))

    y1 = model(torch.tensor([[3.0, 6.0]]))
    assert torch.allclose(y1, torch.Tensor([[3.0]]))
