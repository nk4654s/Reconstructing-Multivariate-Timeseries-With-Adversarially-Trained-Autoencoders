"""The pipeline combines both, the preprocessing.py and postprocessing.py which comes in handy
when processing a dataset using a defined yaml-config. This config should look like:
```
folder: dataset-name
machine-uuid: <UUID>
intervals:
  - <start> <end>
  - ...

preprocessing:
  - class_path: <class-path-with-package>
    init_args:
      <arg-name>: <arg-value>

postprocessing:
  - class_path: <class-path-with-package>
    init_args:
      <arg-name>: <arg-value>
```
"""

import os
import shutil
import logging
from datetime import date
from Dataflux.cli import CLIArgumentParser
from Dataflux.diskio import FileWorkerFactory, FileWorkerPool
from Dataflux.helper.metadata import Metadata
import Dataflux.modules.preprocess as pre
from Dataflux.data import S3, DFIO, BucketType

from Dataflux import get_logger

logger = get_logger(__name__, path="logfile", level=logging.DEBUG)

# define modules here which should always be applied, regardless of the yaml config file!
modules_pre = [
    pre.HomogeneousColumns,
    pre.RenameColumns,
    pre.InterpolatePreProcessor,
    pre.HomogeneousTimeSteps
]

modules_post = []


def preprocessing(parser: CLIArgumentParser) -> None:
    """
    The pre-processing operations. This function initializes the CLIArgumentParser with
    the module classes and handles command-line interface (CLI) arguments.

    Args:
        parser (CLIArgumentParser): The CLIArgumentParser which contains all arguments
            parsed from the command line interface.
    """
    # initialize the pre-processing modules
    modules_pre = parser.get_modules()["preprocessing"]

    # clean up?
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

    s3 = S3(machine_uuid, interval_list)
    s3.set_bucket(BucketType.MEASUREMENTS)

    # download and process data
    dfio = DFIO(s3)
    dfio.download(parser.args.folder, modules=modules_pre)


def postprocessing(parser: CLIArgumentParser) -> None:
    """
    The post-processing operations. This function initializes the CLIArgumentParser
    with the module classes and handles command-line interface (CLI) arguments.

    Args:
        parser (CLIArgumentParser): The CLIArgumentParser which contains all arguments
            parsed from the command line interface.
    """
    # initialize custom CLIArgumentParser
    modules_post = parser.get_modules()["postprocessing"]

    # create multi-threaded file-hanlder
    factory = FileWorkerFactory(parser.args.folder, parser.args.num_workers)
    pool: FileWorkerPool = factory.initialize_workers(modules_post)

    # start processing files and provided file-handler
    pool.ensure_module_dependency(modules_post)
    metadata: Metadata = pool.start_processing()
    metadata.save(parser.args.folder)


def main() -> None:
    """The main entry point for the pipeline which executes the pre-processing
    and post-processing steps."""
    # initialize all modules
    parser = CLIArgumentParser(modules_pre, modules_post)

    # 1. run pre-processing step, which also downloads the dataset
    preprocessing(parser)

    # 2. run post-processing step, which requires the pre-processed data
    postprocessing(parser)


if __name__ == "__main__":
    main()
