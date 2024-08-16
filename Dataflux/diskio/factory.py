"""This module contains the FileWorkerFactory."""
import os
import math
from typing import List
from Dataflux.helper import Meta, Metadata
from .pool import FileWorkerPool
from .worker import FileWorkerThread
from .context import FileWorkerContext
from .callable import FileWorkerCallable
from Dataflux import logger


class FileWorkerFactory:
    """This class is used to create a pool of file workers.
    """
    def __init__(self, base_dir: str = "dataset", num_workers: int = 2) -> None:
        """The FileWorkerFactory initializes a pool of file workers.

        Args:
            base_dir (str): The base directory of the dataset.
            num_workers (int): The number of workers to be created.
        """
        self._context = FileWorkerContext()
        self._metadata = Metadata()
        self._metadata.load(base_dir)

        self._num_workers = num_workers
        self._pool = FileWorkerPool(self._context, self._metadata)

    def initialize_workers(self, modules: List[FileWorkerCallable]) -> FileWorkerPool:
        """The method initializes the workers in the pool and provides a list of files
        to process to each worker.

        Returns:
            FileWorkerPool: The pool of workers that are ready for processing.
        """
        # get all DataFrame files
        path = self._metadata[Meta.PATH]
        files = []
        for f in os.listdir(path):
            if os.path.isdir(os.path.join(path, f)):
                continue
            files.append(os.path.join(path, f))

        # divide list of files into chunks and give one chunk to each worker
        n = len(files) # e.g. 100
        s = max(math.ceil(n / self._num_workers), 1) # e.g. threads=4 -> s=25
        logger.debug("One worker processes %d files", s)
        logger.debug("Using %d workers", self._num_workers)
        for m in modules:
            for i in range(0, n, s): # i=0, i=25, i=75, ...
                logger.debug("Adding files %s to worker %d", files[i:i + s], i)
                worker = FileWorkerThread(files[i:i + s], self._context)
                self._pool.register_worker(worker, m)

        return self._pool
