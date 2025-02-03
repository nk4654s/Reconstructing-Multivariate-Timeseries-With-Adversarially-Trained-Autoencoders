"""Contains the base class for a post-processing module."""
from enum import Enum, unique, auto
from typing import Dict, List, Tuple, Union
from .context import FileWorkerContext
from Dataflux.diskio.meta import FileMeta, Metadata

CALLABLE_RETURN_TYPE = Tuple[Union[FileMeta, List[FileMeta], None], Union[FileMeta, List[FileMeta], None]]


@unique
class CallableDependency(Enum):
    """Represents dependencies for callable objects.
    
    This enum represents different types of dependencies between
    callable objects.
    """
    IMMEDIATELY_AFTER = auto()
    IMMEDIATELY_BEFORE = auto()
    EXISTS = auto()
    NOT_BEFORE = auto()
    BEFORE = auto()
    FINAL = auto()


class FileWorkerCallable:
    """The FileWorkerCallable defines the task, which the
    FileWorkerThread should execute.
    """

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        """This method should return a list of dependencies for this callable instance.

        Returns:
            List[str]: A list of dependencies for this callable instance.
        """
        return {}

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        """This method will be executed on last and should be used to do some clean-up.

        Args:
            context (FileWorkerContext): A context object, which contains
            information passed between modules.
            metadata (Metadata): A Metadata object containing information about
            the dataset.
        """
        raise NotImplementedError

    def run(self, fm: FileMeta, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        """This method will be executed by the FileWorkerThread.

        Args:
            fm (FileMeta): The file information.
            context (FileWorkerContext): A context object, which contains
            information passed between modules.

        Returns:
            File path(s), if the worker module added new files to the dataset.
        """
        raise NotImplementedError
