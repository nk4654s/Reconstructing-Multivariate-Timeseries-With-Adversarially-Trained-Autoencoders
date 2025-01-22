"""This submodule contains the downloading and pre-/postprocessing pipeline.
The pipeline is built generic, as it can be configured by defining
pre-/postprocessing modules. Example code can be found in the entrypoints
`Dataflux.preprocess` and `Dataflux.postprocess`."""

from Dataflux.data.download import (
    DataDownloader,
    ContinuousDataDownloader,
    EmptyStreamError,
)
from Dataflux.data.update import UpdateCallable, DataUpdater, EmptyScoreRegister
from Dataflux.data.types import DataType

__all__ = [
    "DataDownloader",
    "ContinuousDataDownloader",
    "EmptyScoreRegister",
    "EmptyStreamError",
    "UpdateCallable",
    "DataUpdater",
    "DataType",
]
