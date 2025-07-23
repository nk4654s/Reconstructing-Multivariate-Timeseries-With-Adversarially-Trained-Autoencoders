import os

from CompriskV2.model import LinkedParameterModel
from CompriskV2.model.autoregressive import AutoregressiveCertaintyModel
from CompriskV2.model.utils import Checkpoint, OnnxArgs
from CompriskV2.tests.utils import CliContext


def test_checkpoint():
    with CliContext(LinkedParameterModel) as context:
        checkpoint = Checkpoint(context.path)
        assert checkpoint
        assert checkpoint.model


def test_checkpoint_export():
    with CliContext(AutoregressiveCertaintyModel) as context:
        checkpoint = Checkpoint(context.path)
        checkpoint._cli.datamodule._train = context.dataset
        checkpoint.export()

        assert os.path.exists(os.path.join(context.path, "model.onnx"))


def test_onnx_metadata_export():
    with CliContext(AutoregressiveCertaintyModel) as context:
        checkpoint = Checkpoint(context.path)
        checkpoint._cli.datamodule._train = context.dataset
        checkpoint.export()

        args = context.dataset.get_args()
        onnx = OnnxArgs.load_from_model(f"{context.path}/model.onnx")
        assert onnx.machine_uuid == str(context.dataset.UUID)
        assert onnx.input_names == args.names_in
        assert onnx.output_names == args.names_out
        assert onnx.type == args.behavior.value
        assert onnx.version == context.cli.config.seed_everything

        # exported delay is in microseconds (not milliseconds)
        corridor = onnx.value_corridor
        corridor[args.names[0]]["delay"] /= 1000
        assert corridor == context.dataset.metadata.state_alpha
