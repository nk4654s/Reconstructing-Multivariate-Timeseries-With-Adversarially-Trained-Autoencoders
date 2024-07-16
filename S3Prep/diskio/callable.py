from typing import Any

from S3Prep.helper.metadata import Metadata


class FileWorkerCallable:
    """The FileWorkerCallable defines the task, which the
    FileWorkerThread should execute.
    """
    def run(self, file_path: str, metadata: Metadata) -> Any:
        """This method will be executed by the FileWorkerThread.

        Args:
            file_path (str): The file path as string.
            metadata (Metadata): The metadata object, which contains
            information about the whole dataset.
        
        Returns:
            Any result of the file processing task. Can be `None`.
        """
        raise NotImplementedError
