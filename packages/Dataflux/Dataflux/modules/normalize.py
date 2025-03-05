import os
from threading import Lock
from typing import Dict, List
import numpy as np
import pickle

import yaml

from Dataflux import logger
from Dataflux.diskio.meta import FileMeta, Metadata, FileState
from Dataflux.helper import FluxScaler, ScalerMethod
from Dataflux.helper.dataframe_ops import remove_outlier
from Dataflux.diskio import (
    FileWorkerCallable,
    FileWorkerContext,
    CallableDependency,
    CALLABLE_RETURN_TYPE,
)

mutex = Lock()


class ContextParam:
    """Represents context parameters for normalization operations.

    This class provides constants for values and headers used
    in normalization operations.
    """

    SCALER = "normalize-context"


class NormalizeFit(FileWorkerCallable):
    """Represents a fit operation for normalization.

    This class provides methods and constants for performing
    normalization fits. To apply a normalization, use the `NormalizeTransform`
    class.

    Attributes:
        METHODS (list): A list of supported normalization methods. For supported methods, see
            `Dataflux.helper.ScalerMethod`.

    Args:
        method (str): The normalization method to use. Defaults to `minmax`. For supported
            methods, see `Dataflux.helper.ScalerMethod`.
    """

    METHODS = {m.value: m for m in ScalerMethod}

    def __init__(self, method: str = "minmax", remove_outliers: bool = False) -> None:
        assert method in self.METHODS.keys(), (
            f"Scaling method {method} is not supported."
        )
        self._method = self.METHODS[method]
        self._remove = remove_outliers

        logger.debug(
            "Fitting normalizer on dataset, set remove outliers to %s", remove_outliers
        )

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {NormalizeTransform.__name__: CallableDependency.IMMEDIATELY_AFTER}

    def _get_scaler(self, context: FileWorkerContext) -> FluxScaler:
        # get minimum and maximum values from previous files
        scaler = context.get(ContextParam.SCALER)
        if scaler is None:
            scaler = FluxScaler(self._method)

        return scaler

    def run(self, fm: FileMeta, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        # Load the file into a pandas DataFrame, sort by column names to ensure that
        # dataframe is consistent
        df = fm.read()
        df = df.reindex(sorted(df.columns), axis=1)  # type: ignore

        # find and remove outliers
        if self._remove:
            df = remove_outlier(df)
            fm.samples = len(df)
            fm.save(df)

        # get minimum and maximum values from previous files
        scaler = self._get_scaler(context)
        scaler.fit(df)

        context.put(ContextParam.SCALER, scaler)
        return fm

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        logger.debug("Finished fitting the normalizer. Used method %s", self._method)


class NormalizeTransform(FileWorkerCallable):
    """
    Represents a transformation operation for normalization. This class is responsible
    for transforming data into the normalized form.

    It assumes that the `NormalizeFit` class has been run beforehand, and the
    `FileWorkerContext` contains values needed to normalize a dataframe.

    Args:
        from_other (str): Another dataset to use the normalizer from.
    """

    def __init__(self, from_other: str = "", custom: List[str] | None = None) -> None:
        self._scaler = None
        self._columns = list()
        if from_other:
            norm_path = os.path.join(from_other, "metadata", "normalizer.pkl")
            with open(norm_path, "rb") as f:
                self._scaler = pickle.load(f)
                logger.debug("Loaded normalization of other dataset.")

        self._custom = (
            [[e.strip() for e in c.split(",")] for c in custom] if custom else []
        )
        self._custom = sorted(self._custom, key=lambda x: x[0])

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {NormalizeFit.__name__: CallableDependency.IMMEDIATELY_BEFORE}

    def _get_scaler(self, context: FileWorkerContext) -> FluxScaler | None:
        # if the scaler was received, just return it
        if self._scaler:
            return self._scaler

        # get the scaler from the context
        scaler: FluxScaler = context.get(ContextParam.SCALER)
        if not scaler:
            return None
        logger.debug("Got scaler from 'NormalizeFit'.")

        # modify the scaler if a custom one was defined
        if self._custom:
            f_in = sorted(scaler._internal_scaler.feature_names_in_)
            for i, feature in enumerate(f_in):
                name, min_, max_ = self._custom[i]
                assert name == feature, (
                    f"Expected defined name ({name}) to match scaling feature ({feature})"
                )

                scaler.overwrite_column(i, float(min_), float(max_))
            logger.debug("Modified scaler with custom values.")

        self._scaler = scaler
        return scaler

    def run(self, fm: FileMeta, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        # get the used normalization method and the fitted values
        # if there is no scaler, remove all files...
        scaler = self._get_scaler(context)
        if not scaler:
            fm.state = FileState.DELETED
            return fm

        # read the file as pandas dataframe and sort by column names
        # to be consistent with the fitted values
        df = fm.read()
        df = df.reindex(sorted(df.columns), axis=1)  # type: ignore

        # use the scaler to transform the loaded data
        normed = scaler.transform(df, dtype=np.float32)
        fm.save(normed)
        return fm

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        # get the fitted values and method
        scaler: FluxScaler = self._get_scaler(context)

        # clear the context parameters as they are not needed anymore
        context.clear(ContextParam.SCALER)

        # save the normalizer
        norm_path = os.path.join(metadata.path, "metadata", "normalizer.pkl")
        with open(norm_path, "wb") as f:
            pickle.dump(scaler, f)

        # save the normalizer internal data
        path = os.path.join(metadata.path, "metadata", "normalizer.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(scaler.export(), f)

        logger.debug("Finished normalizing the dataset.")
