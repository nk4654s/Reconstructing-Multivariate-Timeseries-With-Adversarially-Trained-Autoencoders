"""This module pre-processes a dataframe and insters values to NAN."""

from typing import Dict, List

import pandas as pd
from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency, CALLABLE_RETURN_TYPE
from Dataflux.helper import Metadata

from Dataflux import logger
from Dataflux.modules.rename import Rename


class Filter(FileWorkerCallable):
    """This class filters according to given sensor borders. The given format must be a dictionary of the format

    `{"name": (string), "lower": (float), "upper": (float)}`

    with lower beeing smaller than higher.
    """

    def __init__(self, filters: List[Dict]):
        super().__init__()

        self._filters = filters
        self._to_remove = []
        logger.debug("Applying filters: %s", str(filters))

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {Rename.__name__: CallableDependency.BEFORE}

    def run(self, file_path: str, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        df = pd.read_feather(file_path)
        
        for filter_dict in self._filters:
            key = filter_dict["name"]
            lower = filter_dict["lower"]
            upper = filter_dict["upper"]

            if df[key].min() < lower or df[key].max() > upper:
                self._to_remove.append(file_path)
                return None, [file_path]
        return None, None
            
    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        pass
