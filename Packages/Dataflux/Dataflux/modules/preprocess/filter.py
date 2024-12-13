"""This module pre-processes a dataframe and insters values to NAN."""

from typing import Dict, List
from Dataflux.modules import BasePreProcessor
from Dataflux.helper import Metadata, DataWrapper
from Dataflux import logger


class Filter(BasePreProcessor):
    """This class filters according to given sensor borders. The given format msut be a dictionary of the format

    `{"name": (string), "lower": (float), "upper": (float)}`

    with lower beeing smaller than higher.
    """

    def __init__(self, filters: List[Dict]):
        super().__init__()

        self._filters = filters
        logger.debug("Applying filters: %s", str(filters))

    def on_download_finished(self, wrapper: DataWrapper, metadata: Metadata) -> List[DataWrapper]:
        for filter_dict in self._filters:
            key = filter_dict["name"]
            lower = filter_dict["lower"]
            upper = filter_dict["upper"]

            if wrapper.df[key].min() < lower or wrapper.df[key].max() > upper:
                return []

        return [wrapper]
