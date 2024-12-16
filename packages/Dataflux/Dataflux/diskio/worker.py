"""This module contains the FileWorkerThread."""
import os
import threading
import time
from typing import List

from .callable import FileWorkerCallable
from .context import FileWorkerContext
from Dataflux import logger


class FileWorkerThread(threading.Thread):
    """The FileWorkerThread enables processing multiple files simultaneosly.
    """
    def __init__(self, module: FileWorkerCallable, files: List[str], context: FileWorkerContext, name: str) -> None:
        """Initializes a Worker object.

        Args:
            files (List[str]): Files to be worked on.
            context (FileWorkerContext): FileWorkerContext to be used by the callable.
        """
        super().__init__(name=name)
        self._files = files
        self._context = context
        self._module = module
    
    @property
    def files(self) -> List[str]:
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
        new_files, del_files = [], []
        for f in self._files:
            if not os.path.exists(f):
                continue

            # make sure this thread does not crash, no matter what the module does
            # if the module contains faulty code, catch the exception and remove the
            # file from the pipeline
            try:
                to_add, to_delete = self._module.run(f, self._context)
            except BaseException as e:
                logger.error("An error occured working on file %s with module %s.", f, self.name)
                logger.error("%s", e)
                to_add, to_delete = [], [f]

            if to_add:
                new_files += to_add if isinstance(to_add, list) else [to_add]
            
            if to_delete:
                del_files += to_delete if isinstance(to_delete, list) else [to_delete]
        
        if len(del_files) > 0:
            logger.debug("Thread of %s deleted %s files", self.name, len(del_files))
            for path in del_files:
                self._files.remove(path)
                if os.path.exists(path):
                    os.remove(path)

        if len(new_files) > 0:
            logger.debug("Thread of %s self-extended by %s files", self.name, len(new_files))
            for path in new_files:
                self.add_file(path)
