"""This module contains the default CLI arguments for the Dataflux pipeline."""

DEFAULT_ARGUMENTS = {
    "folder": "dataset-cat",
    "latest": False,
    "machine_uuid": "0b3426b3-94af-4b8c-89c1-da57493c099a",
    "overwrite": False,
    "disk_backup": None,
    "aws_postfix": "",
    "num_workers": 1,
    "intervals": ["2021-10-15 2021-10-22"]
}
