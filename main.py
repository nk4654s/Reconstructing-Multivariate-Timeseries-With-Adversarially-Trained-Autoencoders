"""The main module handles program execution and parameter modification."""
import os
import shutil
from datetime import date
from S3Prep.cli import MyCLIArgumentParser
from S3Prep.modules import (
    InterpolatePreProcessor,
    MachineStateSplit,
    OutlierPreProcessor,
    Report,
    HomogeneousTimeSteps,
    HomogeneousColumns,
    RemoveColumns,
    RenameColumns,
    CoolingStatus,
    Normalize,
)
from S3Prep.data import S3, DFIO, BucketType


if __name__ == "__main__":
    parser = MyCLIArgumentParser(
        preps=[
            HomogeneousTimeSteps, HomogeneousColumns,
            RenameColumns,# RemoveColumns,
            InterpolatePreProcessor, MachineStateSplit,
            #OutlierPreProcessor,
            Report, CoolingStatus, Normalize,
        ]
    )

    # initialize the pre-processing modules
    prep_modules = parser.get_pre_processors()

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
