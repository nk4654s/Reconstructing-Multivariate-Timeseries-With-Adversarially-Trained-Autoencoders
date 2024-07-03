"""A custom cli interface for simplifying things."""
from typing import List
from S3Prep.helper.cli import CLIArgumentParser
from S3Prep.modules.module import BasePreProcessor


class MyCLIArgumentParser(CLIArgumentParser):
    def __init__(self, preps: List[BasePreProcessor] | BasePreProcessor, *args, **kwargs):
        super().__init__(preps, *args, **kwargs)

    def prepare_arguments(self):
        self.add_argument(
            "--machine-uuid",
            type=str,
            default="0b3426b3-94af-4b8c-89c1-da57493c099a",
            help="Machine UUID"
            )
        self.add_argument("--interval",
                          nargs="+",
                          default=["2020-10-28", "2024-06-30"],
                          help="Interval to get the data from. Format: [YYYY-MM-DD, YYYY-MM-DD]"
                          )
        self.add_argument("--folder",
                          type=str,
                          default=".",
                          help="Folder to save the dataframes in."
                         )
