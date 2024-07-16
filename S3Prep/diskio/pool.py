"""This module contains the FileWorkerPool"""
from typing import List
from S3Prep.diskio import FileWorkerThread, FileWorkerCallable


class FileWorkerPool:
    """This class manages a list of FileWorkerThread objects.
    """
    def __init__(self) -> None:
        self._workers: List[FileWorkerThread] = []

    def register_worker(self, worker: FileWorkerThread):
        """The method registers a new worker to the pool of workers
        """
        assert worker not in self._workers
        self._workers.append(worker)

    def start_processing(self, modules: List[FileWorkerCallable]) -> None:
        """The method starts processing all files of the dataset using the
        registered workers. All workers wait until all files are processed,
        before applying the next module.

        Args:
            modules (List[FileWorkerCallable]): a list of callable objects that
            will be applied to each file.
        """
        for m in modules:
            for w in self._workers:
                w.set_module(m)
                w.start()

            for w in self._workers:
                w.join()
