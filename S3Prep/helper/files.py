"""This module contains file workers."""
import threading
from typing import Any, List


class FileWorkerCallable:
    """The FileWorkerCallable defines the task, which the
    FileWorkerThread should execute.
    """
    def run(self, file_path: str) -> Any:
        """This method will be executed by the FileWorkerThread.

        Args:
            file_path (str): The file path as string.
        
        Returns:
            Any result of the file processing task. Can be `None`.
        """
        raise NotImplementedError


class FileWorkerThread(threading.Thread):
    """The FileWorkerThread enables processing multiple files simultaneosly.
    """
    def __init__(self, files: List[str], process: FileWorkerCallable) -> None:
        """Initializes a Worker object.

        Args:
            files (List[str]): Files to be worked on.
            process (FileWorkerCallable): The callable which operates a file.
        """
        super().__init__()
        self._files = files
        self._process = process
        self._result = []

    def run(self) -> None:
        for f in self._files:
            res = self._process.run(f)
            if res:
                self._result.append(res)
