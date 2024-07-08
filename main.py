"""The main module handles program execution and parameter modification."""

from datetime import date
from S3Prep.helper import MyCLIArgumentParser
from S3Prep.modules import InterpolatePreProcessor, MachineStateSplit, OutlierPreProcessor, Report, HomogeneousTimeSteps, RemoveColumns, Normalize
from S3Prep.data import S3, DFIO


if __name__ == "__main__":
    parser = MyCLIArgumentParser(preps=[HomogeneousTimeSteps,
                                        RemoveColumns,
                                        InterpolatePreProcessor,
                                        MachineStateSplit,
                                        OutlierPreProcessor,
                                        Report,
                                        Normalize])
    prep_modules = parser.get_pre_processors()

    machine_uuid = parser.args.machine_uuid
    interval = (
        date.fromisoformat(parser.args.interval[0]),
        date.fromisoformat(parser.args.interval[-1]),
    )

    s3 = S3(machine_uuid, interval)
    dfio = DFIO(s3, limit_headers=parser.args.limit_headers)
    dfio.download(parser.args.file, modules=prep_modules)
