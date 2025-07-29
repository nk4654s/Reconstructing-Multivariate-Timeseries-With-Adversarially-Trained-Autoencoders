import torch
from CompriskV2.model.ko3 import LinearCalibration


def test_linear_calibration_forward():
    model = LinearCalibration(3)
    x = torch.rand(10, 3)
    y = model(x)
    assert torch.allclose(x, y)


def test_linear_calibration_trainable():
    model = LinearCalibration(3)
    y = model(torch.rand(10, 3))
    assert y.requires_grad


def test_linear_calibration_training_slope():
    model = LinearCalibration(3)
    data = torch.rand(100, 3)
    optim = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.L1Loss()

    for _ in range(50):
        for x in data:
            y = 2 * x
            optim.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y, y_hat)
            loss.backward()
            optim.step()

    assert not torch.allclose(model._slope, torch.ones(3))
    assert torch.allclose(model._slope, torch.tensor([2.0, 2.0, 2.0]), rtol=1e-2)


def test_linear_calibration_training_intercept():
    model = LinearCalibration(3)
    data = torch.rand(100, 3)
    optim = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.L1Loss()

    for _ in range(50):
        for x in data:
            y = x + 2
            optim.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y, y_hat)
            loss.backward()
            optim.step()

    assert torch.all(model._intercept > 0)
    assert torch.allclose(model._intercept, torch.tensor([2.0, 2.0, 2.0]), rtol=1e-2)
