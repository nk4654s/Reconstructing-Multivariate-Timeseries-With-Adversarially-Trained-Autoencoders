import os

from CompriskV2.tests.utils import CliContext
from CompriskV2.model.utils import Checkpoint, ModelIterator
from CompriskV2.model.utils.checkpoint import get_best_checkpoint
from CompriskV2.model.autoregressive import AutoregressiveCertaintyModel


def test_export():
    with CliContext(AutoregressiveCertaintyModel) as context:
        checkpoint = Checkpoint(context.path)
        checkpoint._cli.datamodule._train = context.dataset
        checkpoint.export()

        assert os.path.exists(f"{context.path}/model.onnx")


def test_iterator():
    with CliContext(AutoregressiveCertaintyModel) as context:
        checkpoint = Checkpoint(context.path)
        checkpoint._cli.datamodule._train = context.dataset
        checkpoint.export()

        iterator = ModelIterator(f"{context.path}/model.onnx", context.dataset)
        result = iterator.run()
        assert result


def test_training():
    with CliContext(AutoregressiveCertaintyModel) as context:
        context.cli.trainer.fit(context.cli.model, context.cli.datamodule)
        context.cli.trainer.save_checkpoint(os.path.join(context.ckpt_path, "loss=0.ckpt"))
        assert "loss=0.ckpt" in os.listdir(context.ckpt_path)

        checkpoint = Checkpoint(context.path)
        checkpoint._cli.datamodule._train = context.dataset
        checkpoint.export()

        files = os.listdir(context.ckpt_path)
        ckpt_path = get_best_checkpoint(context.ckpt_path, files)
        assert f"{context.ckpt_path}/loss=0.ckpt" == ckpt_path
