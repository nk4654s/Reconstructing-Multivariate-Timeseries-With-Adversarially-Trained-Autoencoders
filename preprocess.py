"""This module downloads and pre-processes a dataset from a given S3 bucket."""

import os
import shutil
from datetime import date
from S3Prep.cli import PreProcessCLI
from S3Prep.modules.preprocess import *
from S3Prep.data import S3, DFIO, BucketType

# define modules to be applied here
modules = [
    HomogeneousColumns,
    RenameColumns,
    InterpolatePreProcessor,
    HomogeneousTimeSteps,
    MachineStateSplit,
    Report,
]

if __name__ == "__main__":
    # initialize the pre-processing modules
    parser = PreProcessCLI(modules)
    prep_modules = parser.get_modules()

    # define the s3 data source
    machine_uuid = parser.args.machine_uuid
    interval = (
        date.fromisoformat(parser.args.interval[0]),
        date.fromisoformat(parser.args.interval[-1]),
    )
    s3 = S3(machine_uuid, interval)
    s3.set_bucket(BucketType.MEASUREMENTS)

    # clean up?
    if parser.args.overwrite and os.path.exists(parser.args.folder):
        shutil.rmtree(parser.args.folder)

    # download and process data
    dfio = DFIO(s3)
    dfio.download(parser.args.folder, modules=prep_modules)
