"""This module pre-processes a dataframe and insters values to NAN."""

from typing import Dict, List

from Dataflux import logger
from Dataflux.diskio import CALLABLE_RETURN_TYPE, CallableDependency, FileWorkerCallable, FileWorkerContext
from Dataflux.diskio.meta import FileMeta, FileState, Metadata
from Dataflux.modules.rename import Rename


class Filter(FileWorkerCallable):
    """This class filters according to given sensor borders. The given format must be a dictionary of the format

    `{"name": (string), "lower": (float), "upper": (float)}`

    with lower being smaller than higher.
    """

    def __init__(self, filters: List[Dict]):
        super().__init__()

        self._filters = filters
        logger.debug("Applying filters: %s", str(filters))

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {Rename.__name__: CallableDependency.BEFORE}

    def run(self, fm: FileMeta, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        df = fm.read()
        
        for filter_dict in self._filters:
            key = filter_dict["name"]
            lower = filter_dict["lower"]
            upper = filter_dict["upper"]

            # mark the file to be deleted if the filter's boundaries do not fit
            if df[key].min() < lower or df[key].max() > upper:
                fm.state = FileState.DELETED
                return fm

        return fm
            
    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        logger.debug("Filter finished: %s", [n["name"] for n in self._filters])
