"""This module contains the FileWorkerThread."""
import os
import threading
from typing import Any, List

from .callable import FileWorkerCallable
from .context import FileWorkerContext


class FileWorkerThread(threading.Thread):
    """The FileWorkerThread enables processing multiple files simultaneosly.
    """
    def __init__(self, files: List[str], context: FileWorkerContext) -> None:
        """Initializes a Worker object.

        Args:
            files (List[str]): Files to be worked on.
            context (FileWorkerContext): FileWorkerContext to be used by the callable.
        """
        super().__init__()
        self._files = files
        self._context = context

        self._module = None
        self._result = {}

    def set_module(self, module: FileWorkerCallable):
        """Sets the module used by this worker thread.

        Args:
            module (FileWorkerCallable): Module to be used by this worker thread.
        """
        self._module = module

    def get_result(self) -> List[Any]:
        """Returns the result of a certain module.

        Returns:
            List[Any]: Result of a certain module.
        """
        return self._result

    def run(self) -> None:
        # process the files using the current module
        self._result = []
        for f in self._files:
            if not os.path.exists(f):
                continue
            res = self._module.run(f, self._context)
            if res:
                self._result.append(res)
