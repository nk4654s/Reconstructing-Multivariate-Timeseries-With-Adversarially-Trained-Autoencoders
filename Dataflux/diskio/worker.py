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
        assert os.path.exists(path), "Can not add an invalid path to a thread."
        self._files.append(path)

    def run(self) -> None:
        # process the files using the current module
        new_files, del_files = [], []
        for f in self._files:
            if not os.path.exists(f):
                continue

            to_add, to_delete = self._module.run(f, self._context)
            if to_add:
                paths = to_add if isinstance(to_add, list) else [to_add]
                new_files += paths
            
            if to_delete:
                paths = to_delete if isinstance(to_delete, list) else [to_delete]
                for path in to_delete:
                    self._files.remove(path)
                    if os.path.exists(path):
                        os.remove(path)
                    del_files.append(path)
        
        logger.debug("Thread of %s deleted %s files", self._module.__class__.__name__, len(del_files))
        if len(new_files) > 0:
            logger.debug("Thread of %s self-extended by %s files", self._module.__class__.__name__, len(new_files))
            for path in new_files:
                self.add_file(path)
