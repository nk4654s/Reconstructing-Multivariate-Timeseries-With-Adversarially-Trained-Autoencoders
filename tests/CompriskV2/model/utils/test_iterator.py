import os
import torch
import numpy as np
from CompriskV2.model.autoregressive import AutoregressiveCertaintyModel
from CompriskV2.model.utils import ModelIterator, Checkpoint
from CompriskV2.tests.utils import CliContext


def test_iterator():
    with CliContext(AutoregressiveCertaintyModel) as context:
        checkpoint = Checkpoint(context.path)
        checkpoint._cli.datamodule._train = context.dataset
        checkpoint.export()

        model_path = os.path.join(context.path, "model.onnx")
        iterator = ModelIterator(model_path, context.dataset)
        result = iterator.run()

        names_out = context.cli.datamodule.features_out
        assert set(result.names) == set(
            names_out
        ), "Iteration features does not match dataset"
        for name in result.names:
            sensor = result.by_name(name)
            df = sensor.to_dataframe()
            assert len(df) == len(
                context.dataset
            ), "Length of iteration does not match dataset length"

        x, y = context.dataset[0]
        x, y = torch.Tensor(x), torch.Tensor(y)
        x, y = x.unsqueeze(0), y.unsqueeze(0)

        y, _, _, _ = context.cli.model(x, y)
        y_ = result.by_name(names_out[0]).data["y_scaled"][0]
        assert np.allclose(
            y[0][0], y_, rtol=1e-3
        ), "Checkpoint output does not match ONNX output"
