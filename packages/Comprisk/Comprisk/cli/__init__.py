"""
## Overview
This submodule contains the CLI derived from lightning.pytorch.cli.LightningCLI to implement customizations and
automations.
"""
from Comprisk.cli.cli import CompriskCLI
from Comprisk.cli.load import CompriskLoadCLI

__all__ = ["CompriskCLI", "CompriskLoadCLI"]
