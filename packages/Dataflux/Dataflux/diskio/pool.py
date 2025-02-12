"""This module contains the FileWorkerPool"""
import os.path
from typing import List, Dict
from tqdm import tqdm
from .callable import FileWorkerCallable, CallableDependency
from Dataflux.diskio.meta import FileMeta, FileState, Metadata
from .worker import FileWorkerThread
from .context import FileWorkerContext as Context
from Dataflux import logger, DEBUG_MODE


class FileWorkerPool:
    """This class manages a list of FileWorkerThread objects.
    """
    def __init__(self, context: Context, metadata: Metadata) -> None:
        self._modules = []
        self._workers: List[Dict] = []
        self._metadata = metadata
        self._context = context

    def register_worker(self, worker_args: Dict):
        """The method registers a new worker to the pool of workers
        """
        self._workers.append(worker_args)

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
                # check that dependencies is registered as a module
                if module_name not in all_modules.keys():
                    continue

                # assert module_name in all_modules.keys(), f"{module_name} should exist"
                if mode == CallableDependency.IMMEDIATELY_AFTER:
                    j, _ = all_modules[module_name]
                    assert i + 1 == j, f"{module_name} should come immediately after {name}"

                elif mode == CallableDependency.IMMEDIATELY_BEFORE:
                    j, _ = all_modules[module_name]
                    assert i - 1 == j, f"{name} should come immediately before {module_name}"
                
                elif mode == CallableDependency.NOT_BEFORE:
                    j, _ = all_modules[module_name]
                    assert i < j, f"{module_name} should come anytime before {name}"

                elif mode == CallableDependency.BEFORE:
                    j, _ = all_modules[module_name]
                    assert i > j, f"{module_name} should come anytime after {name}"
                
                elif mode == CallableDependency.FINAL:
                    j, _ = all_modules[module_name]
                    assert i == j, f"{module_name} should be the last module"
        
        self._modules = modules

    def start_processing(self) -> Metadata:
        """The method starts processing all files of the dataset using the
        registered workers. All workers wait until all files are processed,
        before applying the next module.
        """
        logger.debug("=" * 100)
        logger.debug("=" + " " * 41 + "Start Processing" + " " * 41 + "=")
        logger.debug("=" * 100)
        with tqdm(total=len(self._modules), desc="Processing Dataset", disable=not DEBUG_MODE) as pbar:
            for module in self._modules:
                module_name = module.__class__.__name__
                pbar.set_description(f"Processing (module={module_name})")

                padding = (100 - len(module_name)) // 2
                logger.debug("=" * padding + f"{module_name}" + "=" * padding)

                active: List[FileWorkerThread] = []
                for i, kwargs in enumerate(self._workers):
                    # modify the thread's arguments to define the correct module to use
                    kwargs["name"] = f"{module.__class__.__name__}_{i}"
                    kwargs["module"] = module
                    
                    thread = FileWorkerThread(**kwargs)
                    thread.start()

                    active.append(thread)

                # get the result of all threads
                for thread in active:
                    thread.join()

                    files: List[FileMeta] = []
                    for f in thread.files:
                        # update metadata, remove deleted files
                        if f.state == FileState.DELETED:
                            fm = self._metadata.files.pop(f.path, None)
                            if fm and os.path.exists(fm.path):
                                os.remove(fm.path)

                        else:
                            self._metadata.files[f.path] = f
                            files.append(f)

                    # update the files as some might be added or deleted by the thread
                    thread_id = int(thread.name.split("_")[-1])
                    self._workers[thread_id]["files"] = files

                # run finish the main thread again
                module.finish(self._context, self._metadata)
                logger.info("Module %s finished processing on all %s threads.", module_name, len(self._workers))
                pbar.update(1)

            pbar.set_description("Processing Done")

        logger.debug("=" * 100)
        logger.debug("=" + " " * 42 + "End Processing" + " " * 42 + "=")
        logger.debug("=" * 100)
        logger.debug("-" * 100)
        return self._metadata
