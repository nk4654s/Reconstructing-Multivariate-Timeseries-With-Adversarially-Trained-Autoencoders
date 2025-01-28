"""This module contains the entrypoints for the **Comprisk** package. The entrypoints are required to train, load, and
analyze a model. Available entrypoints can be found in the package's `setup.py`."""

from Comprisk.entrypoints.base import (
    EntrypointContext,
    Environment,
    Pipeline,
    EntrypointBase,
)
from Comprisk.entrypoints.trainer import TrainerEntrypoint
from Comprisk.entrypoints.scoring import TrainScoreTransformerEntrypoint
from Comprisk.entrypoints.analyzer import AnalyzerEntrypoint
from Comprisk.entrypoints.flux import FluxEntrypoint
from Comprisk.entrypoints.generate import GeneratorEntrypoint

__all__ = [
    "Environment",
    "EntrypointContext",
    "EntrypointBase",
    "Pipeline",
    "AnalyzerEntrypoint",
    "TrainerEntrypoint",
    "TrainScoreTransformerEntrypoint",
    "FluxEntrypoint",
    "GeneratorEntrypoint"
]
