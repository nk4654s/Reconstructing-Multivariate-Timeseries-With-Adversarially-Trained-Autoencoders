import os
import numpy as np

from CompriskV2.model.utils.checkpoint import get_best_checkpoint
from CompriskV2.model.utils import Checkpoint, ModelIterator
from CompriskV2.tests.utils import CliContext
from CompriskV2.model.autoregressive import AutoregressiveAutoencoder


def test_export():
    with CliContext(AutoregressiveAutoencoder) as context:
        checkpoint = Checkpoint(context.path)
        checkpoint._cli.datamodule._train = context.dataset
        checkpoint.export()

        assert os.path.exists(f"{context.path}/model.onnx")


def test_iterator():
    with CliContext(AutoregressiveAutoencoder) as context:
        checkpoint = Checkpoint(context.path)
        checkpoint._cli.datamodule._train = context.dataset
        checkpoint.export()

        iterator = ModelIterator(f"{context.path}/model.onnx", context.dataset)
        result = iterator.run()
        assert result


def test_score_data():
    with CliContext(AutoregressiveAutoencoder) as context:
        args = context.dataset.get_args()

        score = context.cli.model._score_data
        assert not score.ready

        for _, y in context.cli.datamodule.train_dataloader():
            score.record(y)

        weight, total = score.finalize("cpu")
        assert score.ready
        assert len(weight) == args.features_out
        assert len(total) == args.features_out

        context.cli.model._weight = weight
        context.cli.model._total = total
        context.cli.trainer.save_checkpoint(os.path.join(context.ckpt_path, "loss=0.ckpt"))
        model = Checkpoint(context.path).model

        assert np.allclose(weight.detach().numpy(), model._weight.detach().numpy(), rtol=1e-3)
        assert np.allclose(total.detach().numpy(), model._total.detach().numpy(), rtol=1e-3)
        _, y = model.embedding
        score = model.score(y, y * 2)
        assert np.all(score.detach().numpy() > 0)


def test_training():
    with CliContext(AutoregressiveAutoencoder) as context:
        context.cli.trainer.fit(context.cli.model, context.cli.datamodule)
        context.cli.trainer.save_checkpoint(os.path.join(context.ckpt_path, "loss=0.ckpt"))
        assert "loss=0.ckpt" in os.listdir(context.ckpt_path)

        checkpoint = Checkpoint(context.path)
        checkpoint._cli.datamodule._train = context.dataset
        checkpoint.export()

        files = os.listdir(context.ckpt_path)
        ckpt_path = get_best_checkpoint(context.ckpt_path, files)
        assert f"{context.ckpt_path}/loss=0.ckpt" == ckpt_path
