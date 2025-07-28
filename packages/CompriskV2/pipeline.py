from CompriskV2.model import LinkedParameterModel
from CompriskV2.thunder import CompriskCLI, DataModule
from CompriskV2.model.utils import Checkpoint, ModelIterator
from CompriskV2.visualization import Report

from Dataflux import get_logger

logger = get_logger(__name__)


def export():
    cli = CompriskCLI(
        LinkedParameterModel, DataModule, subclass_mode_model=True, run=False
    )

    Checkpoint(cli.root_dir).export()


def visualize():
    cli = CompriskCLI(
        LinkedParameterModel, DataModule, subclass_mode_model=True, run=False
    )
    assert (
        cli.config.report_name
    ), "No report name specified, use '--report-name <name>'"

    iterator = ModelIterator(f"{cli.root_dir}/model.onnx", cli.datamodule.valset)
    result = iterator.run(batch_size=512)

    _ = cli.datamodule.standardizer_params(output=True)
    exclude = [k for k in result.sensors[0].data if k not in ["y_scaled", "y_hat"]]
    logger.info("Ignored keys for revert standardization: %s", exclude)
    result.revert_with(cli.datamodule.scaler, except_keys=exclude)

    Report(cli.root_dir, cli.config.report_name).generate_pdf(result)


def train():
    cli = CompriskCLI(
        LinkedParameterModel, DataModule, subclass_mode_model=True, run=False
    )

    std_inputs = cli.datamodule.standardizer_params()
    std_outputs = cli.datamodule.standardizer_params(output=True)
    cli.model.set_transformers(std_inputs, std_outputs)

    cli.trainer.fit(cli.model, cli.datamodule)
    Checkpoint(cli.trainer.log_dir).export()


if __name__ == "__main__":
    train()
