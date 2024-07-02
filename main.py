"""The main module handles program execution and parameter modification."""
import argparse
from argparse import ArgumentParser
from datetime import date
from s3 import fetch_interval, splice_streams, StorageMode, TsvIO


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--company", type = str, default = "ontras")
    parser.add_argument("--machine-uuid", type = str, default = "0b3426b3-94af-4b8c-89c1-da57493c099a")
    parser.add_argument("--interval", nargs="+", default = ["2020-10-28", "2024-06-30"])
    parser.add_argument("--output-file", "-o", type = str, default = "file.tsv")
    args = parser.parse_args()

    company = args.company
    machine_uuid = args.machine_uuid
    interval = (date.fromisoformat(args.interval[0]),
                date.fromisoformat(args.interval[-1]))

    tsv_readers = fetch_interval(company, machine_uuid, interval)
    tsv_io: TsvIO = splice_streams(tsv_readers, storage_mode=StorageMode.ON_DISK)
    tsv_io.write_to_file(args.output_file)

    assert tsv_io
