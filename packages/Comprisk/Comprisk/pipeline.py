import os.path

from Dataflux import get_logger, logging_mode

from Comprisk.cli import CompriskCLI
from Comprisk.data import DataModule
from Comprisk.helper.structure import get_config_folder
from Comprisk.model import NormalizeWrapper
from Comprisk.entrypoints import (
    EntrypointContext,
    Pipeline,
    TrainerEntrypoint,
    TrainScoreTransformerEntrypoint,
    AnalyzerEntrypoint,
    FluxEntrypoint,
    GeneratorEntrypoint,
    ExportEntrypoint,
)


logger = get_logger(__name__, path="logfile", level=logging_mode)

flux = FluxEntrypoint()
score = TrainScoreTransformerEntrypoint()
train = TrainerEntrypoint()
analyze = AnalyzerEntrypoint()
generate = GeneratorEntrypoint()
export = ExportEntrypoint()


def get_context(cli: CompriskCLI) -> EntrypointContext:
    return EntrypointContext(
        config=cli.config,
        config_folder=get_config_folder(cli),
        model=cli.model,
        datamodule=cli.datamodule,
        trainer=cli.trainer,
        environment_=cli.config.environment,
    )

def pipeline_runner(pipeline: Pipeline, cli: CompriskCLI) -> None:
    for i, name in pipeline.run(context=get_context(cli)):
        logger.info("Finished entrypoint %s (%s)", name, i)


def cli_train() -> None:
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    cli = CompriskCLI(NormalizeWrapper, DataModule, subclass_mode_model=True, run=False)
    pipeline = Pipeline(entrypoints=[
        flux,
        train,
        score,
        export
    ])
    pipeline_runner(pipeline, cli)


def cli_analyze() -> None:
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    cli = CompriskCLI(NormalizeWrapper, DataModule, subclass_mode_model=True, run=False)
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
    cli = CompriskCLI(NormalizeWrapper, DataModule, subclass_mode_model=True, run=False)
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
    cli = CompriskCLI(NormalizeWrapper, DataModule, subclass_mode_model=True, run=False)

    # check if the upload.yaml is in the config folder, otherwise an upload is not possible
    cli.config.data.validate = os.path.join(get_config_folder(cli), "upload.yaml")
    if not os.path.exists(cli.config.data.validate):
        logger.error("Unable to find upload configuration: %s", cli.config.data.validate)
        return

    pipeline = Pipeline(entrypoints=[
        flux,
        generate
    ])
    pipeline_runner(pipeline, cli)


if __name__ == "__main__":
    #cli_train()
    cli_train_score_transformer()
    #cli_analyze()
    #cli_generate()
