"""This module contains the PostProcessingCLI"""

from typing import List
from S3Prep.cli import CLIArgumentParser
from S3Prep.modules import BasePreProcessor


class PostProcessingCLI(CLIArgumentParser):
    """This custom class expands the CLIArgumentParser class."""

    def __init__(
        self, modules: List[BasePreProcessor] | BasePreProcessor, *args, **kwargs
    ):
        super().__init__(modules, *args, **kwargs)

    def prepare_arguments(self):
        self.add_argument(
            "--folder",
            type=str,
            default=".",
            help="Folder path to save the dataframes. Format: [path/to/folder]",
        )
        self.add_argument(
            "--num-workers", type=int, default=2, help="Number of threads to use."
        )
