import os
from threading import Lock
from typing import Any, Callable
import pandas as pd
from Dataflux.helper import Meta, Metadata
from Dataflux.diskio import FileWorkerCallable, FileWorkerContext

mutex = Lock()

class ThreadSafeStore:
    """
    Thread safe store for the HDF5 file.
    """
    def __init__(self, key: str = "data") -> None:
        self._store: pd.HDFStore = None
        self._key = key
        self._path = None

    def is_open(self) -> bool:
        """Returns True if the HDFStore is open."""
        if not self._store:
            return False

        return self._store.is_open

    def close(self) -> None:
        """Closes the HDF5 file."""
        assert self.is_open()
        with mutex:
            self._store.close()
            self._store = None

    def new(self, path) -> None:
        """Opens a new HDF5 file for writing.

        Args:
            path (str): Path to the HDF5 file.
        """
        with mutex:
            if self._store is None:
                self._path = path
                self._store = pd.HDFStore(path, mode="w")
                print("Created Store...")

    def add_dataframe(self, df: pd.DataFrame) -> None:
        """Adds a dataframe to the HDF5 store.
        
        Args:
            df (pd.DataFrame): Dataframe to add to the HDF5 store.
        """
        #assert self.is_open()
        with mutex:
            self._store.put(self._key, df, format="table", append=True)

    def finish(self) -> None:
        """Resets the index `ts` and closes the store.
        """
        with mutex:
            if self._store is not None:
                self._store.close()
                self._store = None

                df = pd.read_hdf(self._path, key=self._key)
                df = df.set_index("ts").reset_index()
                df.to_hdf(self._path, key=self._key, mode="w", format="table")


class MonoFile(FileWorkerCallable):
    """
    Represents a weather dataset for file worker operations. This class provides
    methods and attributes for working with weather data in the context of
    file worker operations.
    """

    def __init__(self, file_name: str = "full", enable: bool = True) -> None:
        self._name = file_name
        self._store = ThreadSafeStore(key="norm")
        self._enable = enable

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        if self._enable is not True or self._enable != "True":
            return None

        print("monolith:", file_path)
        df = pd.read_hdf(file_path, key="norm")
        df = df.set_index("ts").reset_index()

        base_dir = os.path.dirname(file_path)
        self._store.new(f"{base_dir}/{self._name}.h5")
        self._store.add_dataframe(df)

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        if self._enable is not True or self._enable != "True":
            return None
        self._store.finish()
