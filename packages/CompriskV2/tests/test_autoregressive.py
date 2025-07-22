import torch

from CompriskV2.tests.utils.fixtures import CliContext
from CompriskV2.model.autoregressive import AutoregressiveCertaintyModel


def test_model_forward_pass():
    with CliContext(AutoregressiveCertaintyModel) as context:
        model = context.cli.model

        x, y = context.dataset[0]
        x, y = torch.Tensor(x), torch.Tensor(y)
        x, y = x.unsqueeze(0), y.unsqueeze(0)

        y_, xm, xv = model(x, y, return_score=False)
        assert y_.shape == (1, 3)

        y_, xm, xv, sc = model(x, y)
        assert sc.shape == (1, 3)
        assert y_.shape == (1, 3)
