from typing import Any


class FileWorkerContext:
    def __init__(self) -> None:
        self._content = {}

    def put(self, key: str, value: Any) -> None:
        self._content[key] = value

    def append_list(self, key: str, value: Any) -> None:
        l = self._content.get(key, [])
        l.append(value)
        self._content[key] = l

    def get(self, key: str) -> Any:
        return self._content.get(key, None)

    def clear(self, key: str) -> None:
        if self.get(key) is not None:
            del self._content[key]
