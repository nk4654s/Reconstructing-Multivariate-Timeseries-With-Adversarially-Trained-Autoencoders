"""Train a model using a CLI or provide the hyperparameters directly to the CLI."""
from lightning.pytorch.loggers import TensorBoardLogger
from Dataflux.helper import get_logger

# simple demo classes for your convenience
from Comprisk.model import BaseModel
from Comprisk.data import DataModule
from Comprisk.cli import CompriskCLI

logger = get_logger(__name__)


def cli_main():
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    logger.debug("Starting CompriskCLI")
    cli = CompriskCLI(BaseModel, DataModule, subclass_mode_model=True, run=False)

    # set up the logger
    logger.debug("Defining TensorBoard logger for %s", cli.model.__class__.__name__)
    tb_logger = TensorBoardLogger("lightning_logs", name=cli.model.__class__.__name__)
    cli.trainer.logger = tb_logger

    # run methods before fit
    logger.debug("CompriskCLI prepare training.")
    # FIXME: post init using dataset metadata
    cli.prepare()
    cli.before_fit()

    # train
    logger.debug("Starting training")
    cli.trainer.fit(cli.model, cli.datamodule)

    # run methods after fit
    logger.debug("Training finished, clean up now.")
    cli.after_fit()


if __name__ == "__main__":
    cli_main()
