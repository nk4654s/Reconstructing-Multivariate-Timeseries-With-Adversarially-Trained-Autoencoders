from Compmesh.entrypoints.export import SPExportEntrypoint
from Compmesh.entrypoints.trainer import SPTrainEntrypoint
from Comprisk.cli import CompriskCLI
from Comprisk.data import DataModule
from Comprisk.entrypoints import Pipeline
from Comprisk.entrypoints.flux import FluxEntrypoint
from Comprisk.model import NormalizeWrapper
from Comprisk.pipeline import pipeline_runner
from Dataflux import get_logger, logging_mode


logger = get_logger(__name__, path="logfile", level=logging_mode)

flux = FluxEntrypoint()
train = SPTrainEntrypoint()
export = SPExportEntrypoint()#


def cli_train() -> None:
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    cli = CompriskCLI(NormalizeWrapper, DataModule, subclass_mode_model=True, run=False)
    pipeline = Pipeline(entrypoints=[
        flux,
        train,
        export
    ])
    pipeline_runner(pipeline, cli)