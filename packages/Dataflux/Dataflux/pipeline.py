"""The pipeline combines both, the preprocessing.py and postprocessing.py which comes in handy
when processing a dataset using a defined yaml-config.
"""

import os
import shutil
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from Dataflux.cli import CLIArgumentParser
from Dataflux.cli.cli import FluxConfigManager
from Dataflux.diskio import FileWorkerFactory, FileWorkerPool
from Dataflux.helper.metadata import Meta, Metadata
from Dataflux.data import ContinuousDataDownloader, DataType

from Dataflux import logger
from Dataflux.modules import State


modules = []


def download(flux: FluxConfigManager) -> None:
    """
    This function handles the command-line interface (CLI) arguments to download
    and store a dataset.

    Args:
        flux (FluxConfigManager): The CLIArgumentParser which contains all arguments
            parsed from the command line interface.
    """
    # define the s3 data source
    machine_uuid = flux.args.machine_uuid
    interval_list = []
    if flux.args.intervals and not flux.args.latest:
        for interval in flux.args.intervals:
            interval = interval.split()
            start = datetime.fromisoformat(interval[0])

            end = datetime.fromisoformat(interval[1])
            if interval[0] == interval[1]:
                end = end.replace(hour=23, minute=59, second=59)

            interval_list.append((start, end))

    # get the current date and subtract 1 hour to get the model's input
    if flux.args.latest:
        offset = 0
        for m in modules:
            if isinstance(m, State) and m.force_trim:
                offset += m.alpha

        now = datetime.now(tz=timezone.utc)
        past = now - timedelta(seconds=3600 + offset)
        interval_list: List[Tuple[datetime.date, datetime.date]] = [(past, now)]
        past.strftime("%Y-%m-%d %H:%M:%S")

    # exit on wrong inputs
    if not interval_list:
        method = "Latest" if flux.args.latest else "Config Intervals"
        logger.error(
            "Interval list should not be empty. Used method '%s' to get dates.", method
        )
        exit(1)

    logger.info(
        "Downloading interval %s",
        [
            (p.strftime("%Y-%m-%d %H:%M:%S"), n.strftime("%Y-%m-%d %H:%M:%S"))
            for (p, n) in interval_list
        ],
    )
    downloader = ContinuousDataDownloader(machine_uuid, interval_list)
    downloader.set_data_type(DataType.MEASUREMENTS)

    # download and process data
    empty = not downloader.run(flux.args.folder)
    if empty:
        logger.warning("Dataset is empty after download, skipping processing steps.")
        exit(0)


def process(flux: FluxConfigManager) -> None:
    """
    The processing operations. This function initializes the CLIArgumentParser
    with the module classes and handles command-line interface (CLI) arguments.

    Args:
        flux (FluxConfigManager): The CLIArgumentParser which contains all arguments
            parsed from the command line interface.
    """
    # create multithreaded file-handler
    factory = FileWorkerFactory(flux.args.folder, flux.args.num_workers)
    pool: FileWorkerPool = factory.initialize_workers()
    pool.ensure_module_dependency(modules)

    # start processing files and provided file-handler
    metadata: Metadata = pool.start_processing()
    metadata.save(flux.args.folder)


def pipeline(flux: FluxConfigManager) -> str:
    """This function executes the flux pipeline given the FluxConfigManager which holds the required information.

    Args:
        flux (FluxConfigManager): The information about the dataset, e.g. where to download the data and how
            to process them.

    Returns: The output folder.

    """
    global modules
    modules = flux.get_modules()

    if os.path.exists(flux.args.folder) and not flux.args.overwrite:
        logger.warning("Dataset folder exists, overwrite is set to False.")
        exit(0)

    if flux.args.overwrite and os.path.exists(flux.args.folder):
        shutil.rmtree(flux.args.folder)

    # 1. run pre-processing step, which also downloads the dataset
    if flux.args.disk_backup:
        if not os.path.exists(flux.args.disk_backup):
            flux.set_folder(flux.args.disk_backup)
            download(flux)
        shutil.copytree(flux.args.disk_backup, flux.args.folder)
    else:
        download(flux)

    # adjust the metadata, e.g. overwrite the dataset base dir
    metadata = Metadata()
    metadata.load(flux.args.folder)
    metadata.put(flux.args.folder, Meta.PATH)
    metadata.save(flux.args.folder)

    # 2. run post-processing step, which requires the pre-processed data
    process(flux)

    # 3. copy the pipeline configuration used for consistency
    config_path = os.path.join(flux.args.folder, "metadata", "config.yaml")  # type: ignore
    shutil.copyfile(flux.args.config_path, config_path)
    return flux.args.folder


def main() -> None:
    """The main entry point for the pipeline which executes the pre-processing
    and post-processing steps.
    """
    # initialize all modules
    parser = CLIArgumentParser()
    pipeline(parser.get_flux())


if __name__ == "__main__":
    main()
