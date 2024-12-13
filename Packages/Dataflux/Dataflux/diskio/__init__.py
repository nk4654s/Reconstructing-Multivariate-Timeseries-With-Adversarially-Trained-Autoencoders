"""This module contains the threaded super class of postprocessing modules. This
includes the super class itself, as well as callable, dependecies between modules,
a factory to build a module and a pool to execute all build modules. Also, a
context is provided to pass information from one module in the pipeline to a
following one. This comes in handy for training and applying a normalization."""
from Dataflux.diskio.worker import FileWorkerThread
from Dataflux.diskio.callable import FileWorkerCallable, CallableDependency
from Dataflux.diskio.factory import FileWorkerFactory
from Dataflux.diskio.pool import FileWorkerPool
from Dataflux.diskio.context import FileWorkerContext
