"""This module contains the FileWorkerPool"""
from typing import List, Dict
from tqdm import tqdm
from Dataflux.helper import Metadata
from .callable import FileWorkerCallable, CallableDependency
from .worker import FileWorkerThread
from .context import FileWorkerContext


class FileWorkerPool:
    """This class manages a list of FileWorkerThread objects.
    """
    def __init__(self, context: FileWorkerContext, metadata: Metadata) -> None:
        self._modules = []
        self._workers: Dict[str, List[FileWorkerThread]] = {}
        self._metadata = metadata
        self._context = context

    def register_worker(self, worker: FileWorkerThread, module: FileWorkerCallable):
        """The method registers a new worker to the pool of workers
        """
        workers = self._workers.get(module.__class__.__name__, [])
        if len(workers) == 0:
            self._modules.append(module)

        worker.set_module(module)
        workers.append(worker)
        self._workers[module.__class__.__name__] = workers

    def ensure_module_dependency(self, modules: List[FileWorkerCallable]) -> None:
        """The method ensures the dependencies of the modules.

        Args:
            modules (List[FileWorkerCallable]): The list of modules to ensure the dependencies
        """
        all_modules = {m.__class__.__name__: (i, m) for i, m in enumerate(modules)}
        for name, (i, module) in all_modules.items():
            # get dependencies of modules
            dependencies = module.get_dependencies()

            # check registered dependencies
            for module_name, mode in dependencies.items():
                # check that dependecies is registered as a module
                if module_name not in all_modules.keys():
                    continue

                # assert module_name in all_modules.keys(), f"{module_name} should exist"
                if mode == CallableDependency.IMMEDIATELY_AFTER:
                    j, _ = all_modules[module_name]
                    assert i + 1 == j, f"{module_name} should come after {name}"

                elif mode == CallableDependency.IMMEDIATELY_BEFORE:
                    j, _ = all_modules[module_name]
                    assert i - 1 == j, f"{name} should come before {module_name}"
                
                elif mode == CallableDependency.NOT_BEFORE:
                    j, _ = all_modules[module_name]
                    assert i < j, f"{module_name} should come after {name}"
                
                elif mode == CallableDependency.FINAL:
                    j, _ = all_modules[module_name]
                    assert i == j, f"{module_name} should be the last module"

    def start_processing(self) -> Metadata:
        """The method starts processing all files of the dataset using the
        registered workers. All workers wait until all files are processed,
        before applying the next module.
        """
        pbar = tqdm(self._modules, desc="Post-Processing Dataset")
        for module in pbar:
            module_name = module.__class__.__name__
            pbar.set_description(f"Post-Processing Dataset: {module_name}")

            for w in self._workers[module_name]:
                w.start()

            w_results = []
            for w in self._workers[module_name]:
                w.join()
                w_results += w.get_result()

            module.finish(self._context, self._metadata)

        pbar.set_description(f"Post-Processing Dataset: Done")
        return self._metadata
