from enum import Enum, auto
import pandas as pd


class StoreMode(Enum):
    CREATE_OPEN = auto()
    CREATE_CLOSE = auto()
    APPEND_CLOSE = auto()
    APPEND_OPEN = auto()


class DataWrapper:
    def __init__(self, df: pd.DataFrame, path: str, increment: int = -1) -> None:
        self.df = df

        path = path if increment <= 0 else path.replace(".h5", f"_{increment}.h5")
        self._path = path
        self._mode = StoreMode.CREATE_CLOSE
        self._store = None

    @property
    def path(self) -> str:
        return self._path
    
    def set_mode(self, mode: StoreMode) -> None:
        self._mode = mode
    
    def save(self, store: pd.HDFStore | None) -> pd.HDFStore | None:
        match self._mode:
            case StoreMode.CREATE_OPEN:
                if store is not None:
                    store.close()

                store = pd.HDFStore(self._path, "w")
                store.put("data", self.df, format="table", append=True)
                return store
            
            case StoreMode.CREATE_CLOSE:
                if store is not None:
                    store.close()

                store = pd.HDFStore(self._path, "w")
                store.put("data", self.df, format="table", append=True)
                store.close()
                return None
            
            case StoreMode.APPEND_OPEN:
                if store is None:
                    store = pd.HDFStore(self._path, "w")

                store.put("data", self.df, format="table", append=True)
                return store

            case StoreMode.APPEND_CLOSE:
                if store is None:
                    store = pd.HDFStore(self._path, "w")

                store.put("data", self.df, format="table", append=True)
                store.close()
                return None

            case _:
                return None
