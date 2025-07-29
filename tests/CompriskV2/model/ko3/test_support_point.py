import onnxruntime
from CompriskV2.model.utils import Checkpoint
from CompriskV2.tests.utils.fixtures import CliContext
from CompriskV2.model.ko3 import SupportPointModel


def test_support_point_model_loading():
    path = "environment/dev/support_points/gasunie"
    with CliContext(SupportPointModel, model={"ko3_table": path}) as context:
        checkpoint = Checkpoint(context.path)
        checkpoint._cli.datamodule._train = context.dataset
        checkpoint.export()

        session = onnxruntime.InferenceSession(f"{context.path}/support.onnx")
        outputs = [o.shape for o in session.get_outputs()]
        assert outputs[0] == outputs[1] and outputs[1] == outputs[2]
        assert len(outputs[0]) == 3
        assert outputs[0][2] == 3

        context.cli.trainer.fit(context.cli.model, context.cli.datamodule)


def test_thyssen_gas():
    path = "environment/dev/support_points/thyssen_gas"
    with CliContext(SupportPointModel, model={"ko3_table": path}) as context:
        checkpoint = Checkpoint(context.path)
        checkpoint._cli.datamodule._train = context.dataset
        checkpoint.export()

        session = onnxruntime.InferenceSession(f"{context.path}/support.onnx")
        outputs = [o.shape for o in session.get_outputs()]
        assert outputs[0] == outputs[1] and outputs[1] == outputs[2]
        assert len(outputs[0]) == 3
        assert outputs[0][2] == 5

        context.cli.trainer.fit(context.cli.model, context.cli.datamodule)
