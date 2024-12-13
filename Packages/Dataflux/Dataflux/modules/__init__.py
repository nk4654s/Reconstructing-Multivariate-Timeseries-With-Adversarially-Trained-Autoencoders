"""This submodule contains the processing modules applied after
downloading a raw dataset from the Amazon S3 buckt. As some modules require
the whole dataset to be available, there are two types of modules:

- preprocess
- postprocess

The preprocess-modules are called after downloading a file, while the
postprocess-modules are executed on the whole dataset. Hence, the postprocess-modules
inherit from a threaded super class (for more see `Dataflux.diskio`).
"""
from Dataflux.modules.module import BasePreProcessor, EntryPoint
from Dataflux.modules import preprocess, postprocess
