import os
import numpy as np
import torch

from CompriskV2.tests.utils import CliContext
from CompriskV2.model.utils import Checkpoint
from CompriskV2.model import LinkedParameterModel


def test_standardizer():
    with CliContext(LinkedParameterModel) as context:
        model = context.cli.model

        std_inputs = context.cli.datamodule.standardizer_params()
        std_outputs = context.cli.datamodule.standardizer_params(output=True)
        model.set_transformers(std_inputs, std_outputs)

        it = model._input_transformer

        assert std_inputs[0].shape == it._a.shape
        assert std_inputs[1].shape == it._b.shape

        assert np.allclose(std_inputs[0], it._a.detach().numpy(), rtol=1e-3)
        assert np.allclose(std_inputs[1], it._b.detach().numpy(), rtol=1e-3)

        x, y = context.dataset[0]
        x, y = torch.Tensor(x), torch.Tensor(y)
        x, y = x.unsqueeze(0), y.unsqueeze(0)

        x, y = model(x, y, return_score=False)
        assert not x.requires_grad
        assert not y.requires_grad


def test_standardizer_export():
    with CliContext(LinkedParameterModel) as context:
        model = context.cli.model
        std_inputs = context.cli.datamodule.standardizer_params()
        std_outputs = context.cli.datamodule.standardizer_params(output=True)
        model.set_transformers(std_inputs, std_outputs)

        it = model._input_transformer

        context.cli.trainer.save_checkpoint(
            os.path.join(context.ckpt_path, "loss=0.ckpt")
        )
        checkpoint = Checkpoint(context.path)
        model = checkpoint.model

        assert np.allclose(std_inputs[0], it._a.detach().numpy(), rtol=1e-3)
        assert np.allclose(std_inputs[1], it._b.detach().numpy(), rtol=1e-3)
