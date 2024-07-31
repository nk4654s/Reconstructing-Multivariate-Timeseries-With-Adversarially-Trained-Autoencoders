"""Train a model using a CLI or provide the hyperparameters directly to the CLI."""
from Dataflux.helper import get_logger
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

    _ = cli.model.summary()


if __name__ == "__main__":
    cli_main()
