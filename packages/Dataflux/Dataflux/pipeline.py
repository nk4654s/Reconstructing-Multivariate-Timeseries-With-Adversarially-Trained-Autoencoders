"""The pipeline combines both, the preprocessing.py and postprocessing.py which comes in handy
when processing a dataset using a defined yaml-config.
"""
import os
import shutil
import logging
from datetime import datetime, timedelta, timezone
from typing import List
from Dataflux.cli import CLIArgumentParser
from Dataflux.diskio import FileWorkerFactory, FileWorkerPool
from Dataflux.helper.metadata import Meta, Metadata
from Dataflux.data import ContinuousDataDownloader, DataType

from Dataflux import get_logger

logger = get_logger(__name__, path="logfile", level=logging.DEBUG)

# define modules here which should always be applied, regardless of the yaml config file!
modules = []


def download(parser: CLIArgumentParser, folder: str) -> None:
    """
    This function handles the command-line interface (CLI) arguments to download
    and store a dataset.

    Args:
        parser (CLIArgumentParser): The CLIArgumentParser which contains all arguments
            parsed from the command line interface.
        folder (str): The output folder of the dataset.
    """
    # define the s3 data source
    machine_uuid = parser.args.machine_uuid
    interval_list = []
    if parser.args.intervals and not parser.args.latest:
        for interval in parser.args.intervals:
            interval = interval.split(" ")
            interval_list.append((
                datetime.fromisoformat(interval[0]),
                datetime.fromisoformat(interval[1])
            ))

    # get the current date and subtract 1 hour to get the model's input
    if parser.args.latest:
        now = datetime.now(tz=timezone.utc)
        past = now - timedelta(hours=1)
        interval_list = [(past, now)]

    # exit on wrong inputs
    if not interval_list:
        method = "Latest" if parser.args.latest else "Config Intervals"
        logger.error("Interval list should not be empty. Used method '%s' to get dates.", method)
        exit(1)

    downloader = ContinuousDataDownloader(machine_uuid, interval_list)
    downloader.set_data_type(DataType.MEASUREMENTS)

    # download and process data
    downloader.run(folder)


def process(parser: CLIArgumentParser, modules: List) -> None:
    """
    The processing operations. This function initializes the CLIArgumentParser
    with the module classes and handles command-line interface (CLI) arguments.

    Args:
        parser (CLIArgumentParser): The CLIArgumentParser which contains all arguments
            parsed from the command line interface.
        modules (List): The list of classes which should be used for
            processing.
    """
    # create multi-threaded file-hanlder
    factory = FileWorkerFactory(parser.args.folder, parser.args.num_workers)
    pool: FileWorkerPool = factory.initialize_workers()
    pool.ensure_module_dependency(modules)

    # start processing files and provided file-handler
    metadata: Metadata = pool.start_processing()
    metadata.save(parser.args.folder)


def main() -> None:
    """The main entry point for the pipeline which executes the pre-processing
    and post-processing steps."""
    global modules

    # initialize all modules
    parser = CLIArgumentParser(modules)
    modules = parser.get_modules()
    if os.path.exists(parser.args.folder) and not parser.args.overwrite:
        logger.debug("Dataset folder exists, overwrite is set to False.")
        exit(0)

    if parser.args.overwrite and os.path.exists(parser.args.folder):
        shutil.rmtree(parser.args.folder)

    # 1. run pre-processing step, which also downloads the dataset
    if parser.args.disk_backup:
        if not os.path.exists(parser.args.disk_backup):
            download(parser, parser.args.disk_backup)
        shutil.copytree(parser.args.disk_backup, parser.args.folder)
    else:
        download(parser, parser.args.folder)  

    # adjust the metadata, e.g. overwrite the dataset base dir
    metadata = Metadata()
    metadata.load(parser.args.folder)
    metadata.put(parser.args.folder, Meta.PATH)
    metadata.save(parser.args.folder)      

    # 2. run post-processing step, which requires the pre-processed data
    process(parser, modules) 

    # 3. copy the pipeline configuration used for consistency
    config_path = os.path.join(parser.args.folder, "metadata", "config.yaml") # type: ignore
    shutil.copyfile(parser.args.config, config_path)


if __name__ == "__main__":
    main()
