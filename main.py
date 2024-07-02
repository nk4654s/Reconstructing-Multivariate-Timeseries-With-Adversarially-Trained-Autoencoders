"""The main module handles program execution and parameter modification."""

from datetime import date
from helper import CLIArgumentParser
from prep import InterpolatePreProcessor
from data import S3, DFIO


if __name__ == "__main__":
    parser = CLIArgumentParser([InterpolatePreProcessor])
    prep_modules = parser.get_pre_processors()

    machine_uuid = parser.args.machine_uuid
    interval = (
        date.fromisoformat(parser.args.interval[0]),
        date.fromisoformat(parser.args.interval[-1]),
    )

    s3 = S3(machine_uuid, interval)
    dfio = DFIO(s3)
    dfio.download(prep_modules)
