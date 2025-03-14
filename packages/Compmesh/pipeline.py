import os

from Compmesh.entrypoints.export import SPExportEntrypoint
from Compmesh.entrypoints.trainer import SPTrainEntrypoint
from Compmesh.cli import CompmeshCLI
from Comprisk.data import DataModule
from Comprisk.entrypoints import Pipeline, AnalyzerEntrypoint
from Comprisk.entrypoints.flux import FluxEntrypoint
from Comprisk.model import NormalizeWrapper
from Comprisk.pipeline import pipeline_runner
from Dataflux import get_logger, logging_mode


logger = get_logger(__name__, path="logfile", level=logging_mode)

flux = FluxEntrypoint()
train = SPTrainEntrypoint()
export = SPExportEntrypoint()
analyzer = AnalyzerEntrypoint(default_tag="analysis/support_points")


def cli_train() -> None:
    """
    The main entry point for the Compmesh CLI. This function initializes the
    CompmeshCLI object and sets up the command-line interface.
    """
    cli = CompmeshCLI(NormalizeWrapper, DataModule, subclass_mode_model=True, run=False)
    pipeline = Pipeline(entrypoints=[
        flux,
        train,
        export,
        analyzer
    ])
    pipeline_runner(pipeline, cli)

if __name__ == "__main__":
    pycharm_entry = os.environ.get("PYCHARM_ENTRY", "train")
    if pycharm_entry == "train":
        cli_train()
