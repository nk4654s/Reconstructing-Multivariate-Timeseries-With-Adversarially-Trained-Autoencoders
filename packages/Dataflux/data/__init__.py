"""This submodule contains the downloading and pre-/postprocessing pipeline.
The pipeline is built generic, as it can be configured by defining
pre-/postprocessing modules. Example code can be found in the entrypoints
`Dataflux.preprocess` and `Dataflux.postprocess`."""

from Dataflux.data.download import (
    ContinuousDataDownloader,
    DataDownloader,
    EmptyStreamError,
)
from Dataflux.data.types import DataType
from Dataflux.data.update import DataUpdater, EmptyScoreRegister, UpdateCallable

__all__ = [
    "DataDownloader",
    "ContinuousDataDownloader",
    "EmptyScoreRegister",
    "EmptyStreamError",
    "UpdateCallable",
    "DataUpdater",
    "DataType",
]
