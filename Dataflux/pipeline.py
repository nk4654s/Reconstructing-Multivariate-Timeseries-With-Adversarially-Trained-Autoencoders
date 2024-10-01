"""The pipeline combines both, the preprocessing.py and postprocessing.py which comes in handy
when processing a dataset using a defined yaml-config.
"""

import os
import shutil
import logging
from datetime import date
from Dataflux.cli import CLIArgumentParser, TYPE_MODULES_PRE, TYPE_MODULES_POST
from Dataflux.diskio import FileWorkerFactory, FileWorkerPool
from Dataflux.helper.metadata import Metadata
import Dataflux.modules.preprocess as pre
from Dataflux.data import download, ContinuousDataDownloader, DataType

from Dataflux import get_logger

logger = get_logger(__name__, path="logfile", level=logging.DEBUG)

# define modules here which should always be applied, regardless of the yaml config file!
modules_pre = [
    pre.HomogeneousColumns,
    pre.RenameColumns,
    pre.InterpolatePreProcessor,
    pre.HomogeneousTimeSteps,
]

modules_post = []


def preprocessing(parser: CLIArgumentParser, modules: TYPE_MODULES_PRE) -> None:
    """
    The pre-processing operations. This function initializes the CLIArgumentParser with
    the module classes and handles command-line interface (CLI) arguments.

    Args:
        parser (CLIArgumentParser): The CLIArgumentParser which contains all arguments
            parsed from the command line interface.
        modules (TYPE_MODULES_PRE): The list of classes which should be used for
            preprocessing.
    """
    if parser.args.overwrite and os.path.exists(parser.args.folder):
        shutil.rmtree(parser.args.folder)

    # define the s3 data source
    machine_uuid = parser.args.machine_uuid
    interval_list = []
    for interval in parser.args.intervals:
        interval = interval.split(" ")
        interval_list.append((
            date.fromisoformat(interval[0]),
            date.fromisoformat(interval[1])
        ))

    downloader = ContinuousDataDownloader(machine_uuid, interval_list)
    downloader.set_data_type(DataType.MEASUREMENTS)

    # download and process data
    download(downloader, parser.args.folder, modules=modules)


def postprocessing(parser: CLIArgumentParser, modules: TYPE_MODULES_POST) -> None:
    """
    The post-processing operations. This function initializes the CLIArgumentParser
    with the module classes and handles command-line interface (CLI) arguments.

    Args:
        parser (CLIArgumentParser): The CLIArgumentParser which contains all arguments
            parsed from the command line interface.
        modules (TYPE_MODULES_POST): The list of classes which should be used for
            postprocessing.
    """
    # create multi-threaded file-hanlder
    factory = FileWorkerFactory(parser.args.folder, parser.args.num_workers)
    pool: FileWorkerPool = factory.initialize_workers(modules)

    # start processing files and provided file-handler
    pool.ensure_module_dependency(modules)
    metadata: Metadata = pool.start_processing()
    metadata.save(parser.args.folder)


def main() -> None:
    """The main entry point for the pipeline which executes the pre-processing
    and post-processing steps."""
    # initialize all modules
    parser = CLIArgumentParser(modules_pre, modules_post)
    modules = parser.get_modules()

    # 1. run pre-processing step, which also downloads the dataset
    preprocessing(parser, modules[0])

    # 2. run post-processing step, which requires the pre-processed data
    postprocessing(parser, modules[1])

    # 3. copy the pipeline configuration used for consistency
    config_path = os.path.join(parser.args.folder, "metadata", "config.yaml")
    shutil.copyfile(parser.args.config, config_path)


if __name__ == "__main__":
    main()
