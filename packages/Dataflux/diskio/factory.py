"""This module contains the FileWorkerFactory."""

import math
import os

from Dataflux import logger
from Dataflux.diskio.meta import FileMeta, Metadata

from .context import FileWorkerContext as Context
from .pool import FileWorkerPool


class FileWorkerFactory:
    """This class is used to create a pool of file workers."""

    def __init__(self, base_dir: str = "dataset", num_workers: int = 1) -> None:
        """The FileWorkerFactory initializes a pool of file workers.

        Args:
            base_dir (str): The base directory of the dataset.
            num_workers (int): The number of workers to be created.
        """
        self._base_dir = base_dir
        self._metadata = Metadata.load(base_dir)

        self._context = Context()
        self._context.put(Context.DIRECTORY, base_dir)
        self._context.put(Context.UUID, self._metadata.uuid)

        self._num_workers = num_workers
        self._pool = FileWorkerPool(self._context, self._metadata)

    def initialize_workers(self) -> FileWorkerPool:
        """The method initializes the workers in the pool and provides a list of files
        to process to each worker.

        Returns:
            FileWorkerPool: The pool of workers that are ready for processing.
        """
        # get all DataFrame files
        path = self._base_dir
        files = []
        for f in os.listdir(path):
            if os.path.isdir(os.path.join(path, f)):
                continue
            files.append(os.path.join(path, f))

        # divide list of files into chunks and give one chunk to each worker
        n = len(files)  # e.g. 100
        s = max(math.ceil(n / self._num_workers), 1)  # e.g. threads=4 -> s=25
        logger.debug("One worker processes %d files", s)
        logger.debug("Using %d workers", self._num_workers)
        for i in range(0, n, s):  # i=0, i=25, i=75, ...
            worker_files = []
            for f in files[i:i+s]:
                base_path = os.path.dirname(f)
                file_name = os.path.basename(f)
                worker_files.append(FileMeta(file_name, base_path=base_path))

            worker = {
                "files": worker_files,
                "context": self._context,
            }
            self._pool.register_worker(worker)

        logger.debug("Finished worker initialization for %s files", n)
        return self._pool
