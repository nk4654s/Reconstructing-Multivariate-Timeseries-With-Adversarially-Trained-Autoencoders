"""This module contains the FileWorkerContext which may be used to transfer values
and object from one post-processing module to the other."""
from typing import Any
from enum import auto


class FileWorkerContext:
    """Represents the context for file worker operations.
    
    This class provides methods for storing and retrieving data related to
    file worker operations.
    """

    DIRECTORY = auto()
    DELETED_FILES = auto()
    ADDED_FILES = auto()
    UUID = auto()

    def __init__(self) -> None:
        """Initializes the FileWorkerContext with an empty dictionary."""
        self._content = {}

    def put(self, key: Any, value: Any) -> None: # type: ignore
        """
        Stores a key-value pair in the context.

        Args:
            key (Any): The key for the stored value.
            value (Any): The value to be stored.
        """
        self._content[key] = value

    def append_list(self, key: Any, value: Any) -> None:
        """
        Appends a new element to a list stored in the context.

        Args:
            key (str): The key for the list that will have a new element appended.
            value (Any): The new element to be appended to the list.
        """
        container = self._content.get(key, [])
        if isinstance(value, list):
            container += value
        else:
            container.append(value)
        self._content[key] = container

    def get(self, key: Any) -> Any:
        """
        Retrieves the value associated with a given key from the context.

        Args:
            key (Any): The key for which the corresponding value is to be retrieved.

        Returns:
            Any: The value associated with the given key.
        """
        return self._content.get(key, None)

    def clear(self, key: Any) -> None:
        """
        Removes the key-value pair from the context if it exists.

        Args:
            key (Any): The key of the key-value pair to be removed.
        """
        if self.get(key) is not None:
            del self._content[key]
