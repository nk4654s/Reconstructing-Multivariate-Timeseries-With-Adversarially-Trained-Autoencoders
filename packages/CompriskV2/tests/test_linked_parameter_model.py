import numpy as np
import torch

from CompriskV2.tests.utils.fixtures import CliContext
from CompriskV2.model import LinkedParameterModel


def test_standardizer():
    with CliContext(LinkedParameterModel) as context:
        model = context.cli.model

        mean, scale = context.cli.datamodule.standardizer_params()
        model.set_standardizer(mean, scale)

        assert mean.shape == model._mean.shape
        assert scale.shape == model._scale.shape

        assert np.allclose(mean, model._mean.detach().numpy(), rtol=1e-3)
        assert np.allclose(scale, model._scale.detach().numpy(), rtol=1e-3)

        x, y = context.dataset[0]
        x, y = torch.Tensor(x), torch.Tensor(y)
        x, y = x.unsqueeze(0), y.unsqueeze(0)

        x, y = model(x, y, return_score=False)
        assert not x.requires_grad
        assert not y.requires_grad
