"""The main module handles program execution and parameter modification."""

from argparse import ArgumentParser
from datetime import date
from s3 import S3, DFIO


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--machine-uuid", type=str, default="0b3426b3-94af-4b8c-89c1-da57493c099a"
    )
    parser.add_argument("--interval", nargs="+", default=["2020-10-28", "2024-06-30"])
    parser.add_argument("--output-file", "-o", type=str, default="file.tsv")
    args = parser.parse_args()

    machine_uuid = args.machine_uuid
    interval = (
        date.fromisoformat(args.interval[0]),
        date.fromisoformat(args.interval[-1]),
    )

    s3 = S3(machine_uuid, interval)
    dfio = DFIO(s3)
    dfio.download()
