"""
## Overview
This module defines the entrypoints packed together to build a pipeline. Each entrypoint module implements unique
tasks, such as downloading and processing a dataset, training the model, or running evaluations. A pipeline combines
these module in a way to form an executable pipeline. A pipeline can be run using the CLI after installing `Comprisk`.
To learn more about the generic entrypoint structure check out `Comprisk.entrypoints.base` for more information.
"""

from Comprisk.entrypoints.base import (
    EntrypointBase,
    EntrypointContext,
    Environment,
    Pipeline,
)
from Comprisk.entrypoints.trainer import TrainerEntrypoint
from Comprisk.entrypoints.scoring import TrainScoreTransformerEntrypoint
from Comprisk.entrypoints.analyzer import AnalyzerEntrypoint
from Comprisk.entrypoints.flux import FluxEntrypoint
from Comprisk.entrypoints.generate import GeneratorEntrypoint
from Comprisk.entrypoints.export import ExportEntrypoint

__all__ = [
    "EntrypointContext",
    "EntrypointBase",
    "Pipeline",
    "AnalyzerEntrypoint",
    "ExportEntrypoint",
    "GeneratorEntrypoint",
    "FluxEntrypoint",
    "TrainerEntrypoint",
    "TrainScoreTransformerEntrypoint",
    "Environment",
]
