"""The main module handles program execution and parameter modification."""
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
    Normalize,
)
from S3Prep.data import S3, DFIO


if __name__ == "__main__":
    parser = MyCLIArgumentParser(
        preps=[
            HomogeneousTimeSteps, HomogeneousColumns,
            RenameColumns, RemoveColumns,
            InterpolatePreProcessor, MachineStateSplit,
            OutlierPreProcessor, Report, Normalize,
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

    # clean up?
    if parser.args.overwrite:
        shutil.rmtree(parser.args.folder)

    # download and process data
    dfio = DFIO(s3, limit_headers=parser.args.limit_headers)
    dfio.download(parser.args.folder, modules=prep_modules)
