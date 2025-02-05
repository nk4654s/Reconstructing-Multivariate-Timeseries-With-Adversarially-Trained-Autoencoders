"""This submodule contains the processing modules applied after
downloading a raw dataset from the Amazon S3 buckt, as some modules require
the whole dataset to be available.

The process-modules are called after downloading the whole dataset. Hence, the 
modules inherit from a threaded super class (for more see `Dataflux.diskio`).
"""
from Dataflux.modules.rename import Rename
from Dataflux.modules.filter import Filter
from Dataflux.modules.state import State
from Dataflux.modules.slice import Slice


from Dataflux.modules.features import Features
from Dataflux.modules.normalize import NormalizeFit, NormalizeTransform

__all__ = [
    "Rename",
    "Filter",
    "State",
    "Slice",
    "Features",
    "NormalizeFit",
    "NormalizeTransform"
]
