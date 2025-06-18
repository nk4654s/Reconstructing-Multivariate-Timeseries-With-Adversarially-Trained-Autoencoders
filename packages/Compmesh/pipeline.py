import os

from Compmesh.entrypoints.trainer import SPTrainEntrypoint
from Compmesh.cli import CompmeshCLI
from Comprisk.custom import DataModule
from Comprisk.entrypoints import (
    Pipeline,
    AnalyzerEntrypoint,
    TrainScoreTransformerEntrypoint,
    ExportEntrypoint
)
from Comprisk.entrypoints.flux import FluxEntrypoint
from Comprisk.model import NormalizeWrapper
from Comprisk.pipeline import pipeline_runner
from Dataflux import get_logger, logging_mode


logger = get_logger(__name__, path="logfile", level=logging_mode)

flux = FluxEntrypoint()
train = SPTrainEntrypoint()
score = TrainScoreTransformerEntrypoint(reduced_mode=False)
export = ExportEntrypoint(name="support")
analyzer = AnalyzerEntrypoint(default_tag="analysis/support_points", model_name="support")


def cli_train() -> None:
    """
    The main entry point for the Compmesh CLI. This function initializes the
    CompmeshCLI object and sets up the command-line interface.
    """
    cli = CompmeshCLI(NormalizeWrapper, DataModule, subclass_mode_model=True, run=False)
    pipeline = Pipeline(entrypoints=[
        flux,
        train,
        score,
        export,
        analyzer
    ])
    pipeline_runner(pipeline, cli)

def cli_score_trainer() -> None:
    """
    The main entry point for the Compmesh CLI. This function initializes the
    CompmeshCLI object and sets up the command-line interface.
    """
    cli = CompmeshCLI(NormalizeWrapper, DataModule, subclass_mode_model=True, run=False)
    pipeline = Pipeline(entrypoints=[
        flux,
        score,
        export,
        analyzer
    ])
    pipeline_runner(pipeline, cli)

def cli_analyze() -> None:
    """
    The main entry point for the Compmesh CLI. This function initializes the
    CompmeshCLI object and sets up the command-line interface.
    """
    cli = CompmeshCLI(NormalizeWrapper, DataModule, subclass_mode_model=True, run=False)
    pipeline = Pipeline(entrypoints=[
        flux,
        analyzer
    ])
    pipeline_runner(pipeline, cli)

if __name__ == "__main__":
    pycharm_entry = os.environ.get("PYCHARM_ENTRY", "train")
    if pycharm_entry == "train":
        cli_train()
    if pycharm_entry == "analyze":
        cli_analyze()
    if pycharm_entry == "score":
        cli_score_trainer()
