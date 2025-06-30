import os.path

from Comprisk.helper.structure import get_config_folder
from Dataflux import get_logger, logging_mode

from Comprisk.custom import CompriskCLI
from Comprisk.entrypoints.utils import EntrypointContext
from Comprisk.entrypoints import (
    AnalyzerEntrypoint,
    ExportEntrypoint,
    FluxEntrypoint,
    GeneratorEntrypoint,
    Pipeline,
    TrainerEntrypoint,
    TrainScoreTransformerEntrypoint,
    TuneEntrypoint
)

logger = get_logger(__name__, path="logfile", level=logging_mode)

flux = FluxEntrypoint()
score = TrainScoreTransformerEntrypoint()
train = TrainerEntrypoint()
analyze = AnalyzerEntrypoint()
generate = GeneratorEntrypoint()
export = ExportEntrypoint()
tune = TuneEntrypoint()


def pipeline_runner(pipeline: Pipeline, cli: CompriskCLI) -> None:
    context = EntrypointContext(
        config=cli.config,
        config_folder=get_config_folder(cli),
        model=cli.model,
        datamodule=cli.datamodule,
        trainer=cli.trainer,
        environment_=cli.config.environment,
    )

    for i, name in pipeline.run(context=context):
        logger.info("Finished entrypoint %s (%s)", name, i)


def cli_train() -> None:
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    cli = CompriskCLI(subclass_mode_model=True, run=False)
    pipeline = Pipeline(entrypoints=[
        flux,
        train,
        score,
        export,
        analyze
    ])
    pipeline_runner(pipeline, cli)


def cli_tune() -> None:
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    cli = CompriskCLI(subclass_mode_model=True, run=False)
    pipeline = Pipeline(entrypoints=[
        flux,
        tune,
        score,
        export,
        analyze
    ])
    pipeline_runner(pipeline, cli)


def cli_analyze() -> None:
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    cli = CompriskCLI(subclass_mode_model=True, run=False)
    pipeline = Pipeline(entrypoints=[
        flux,
        analyze
    ])
    pipeline_runner(pipeline, cli)


def cli_train_score_transformer() -> None:
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    cli = CompriskCLI(subclass_mode_model=True, run=False)
    pipeline = Pipeline(entrypoints=[
        flux,
        score,
        export
    ])
    pipeline_runner(pipeline, cli)


def cli_generate() -> None:
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    cli = CompriskCLI(subclass_mode_model=True, run=False)
    pipeline = Pipeline(entrypoints=[
        flux,
        generate
    ])
    pipeline_runner(pipeline, cli)


def cli_flux() -> None:
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    cli = CompriskCLI(subclass_mode_model=True, run=False)
    pipeline = Pipeline(entrypoints=[
        flux
    ])
    pipeline_runner(pipeline, cli)


if __name__ == "__main__":
    pycharm_entry = os.environ.get("PYCHARM_ENTRY", "train")
    if pycharm_entry == "train":
        cli_train()
    elif pycharm_entry == "score":
        cli_train_score_transformer()
    elif pycharm_entry == "analyze":
        cli_analyze()
    elif pycharm_entry == "generate":
        cli_generate()
    elif pycharm_entry == "tune":
        cli_tune()
