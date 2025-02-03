"""This module contains the FileWorkerThread."""
import os
import threading
from typing import List

from .callable import FileWorkerCallable
from .context import FileWorkerContext
from Dataflux import logger
from Dataflux.diskio.meta.file_meta import FileMeta, FileState

mutex = threading.Lock()


class FileWorkerThread(threading.Thread):
    """The FileWorkerThread enables processing multiple files simultaneously.
    """
    def __init__(self, module: FileWorkerCallable, files: List[FileMeta], context: FileWorkerContext, name: str) -> None:
        """Initializes a Worker object.

        Args:
            files (List[str]): Files to be worked on.
            context (FileWorkerContext): FileWorkerContext to be used by the callable.
        """
        super().__init__(name=name)
        self._files: List[FileMeta] = files
        self._module = module

        # context is shared over self and other threads
        self._context = context

    @property
    def files(self) -> List[FileMeta]:
        return self._files
    
    def add_file(self, path: str) -> None:
        """Adds a new file to be processed by this thread.
        
        Args:
            path (str): Path to file.
        """
        if not os.path.exists(path):
            logger.debug("Can not add an invalid path to a thread (%s).", path)
            return
    
        self._files.append(path)

    def run(self) -> None:
        logger.debug("Thread %s processes %s files", self.name, len(self._files))
        # process the files using the current module
        add_files = []
        for f in self._files:
            if not os.path.exists(f):
                continue

            # make sure this thread does not crash, no matter what the module does
            # if the module contains faulty code, catch the exception and remove the
            # file from the pipeline
            try:
                add = self._module.run(f, self._context)
                add_files += add if add else []
            except BaseException as e:
                logger.error("An error occurred working on file %s with module %s.", f, self.name)
                logger.error("%s", e)
                f.state = FileState.DELETED

        self._files += add_files
