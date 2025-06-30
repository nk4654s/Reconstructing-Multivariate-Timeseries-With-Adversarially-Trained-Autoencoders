from Comprisk.entrypoints.utils import EntrypointContext


class EntrypointBase:
    def __init__(self) -> None:
        pass

    def __call__(self, context: EntrypointContext, *args, **kwargs) -> EntrypointContext:
        return context


class Pipeline:
    def __init__(self, entrypoints: list[EntrypointBase]) -> None:
        self._entrypoints = entrypoints

    def run(self, context: EntrypointContext) -> tuple[int, str]:
        for i, entry in enumerate(self._entrypoints):
            context: EntrypointContext = entry(context)
            yield i, entry.__class__.__name__
