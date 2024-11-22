import os
from threading import Lock
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from scipy import stats
import pickle

from Dataflux import logger
from Dataflux.helper import Meta, Metadata, FluxScaler, ScalerMethod
from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency, CALLABLE_RETURN_TYPE

mutex = Lock()


def remove_outlier(df: pd.DataFrame) -> pd.DataFrame:
    indicies = []
    data = np.random.normal(loc=0, scale=1, size=100)
    for key in df.columns.values:
        series = df[key].to_numpy()

        # get boxplot length of data series
        q75, q25 = np.percentile(series, [75, 25])
        box_length = q75 - q25

        # calculate the outlier boundaries of the boxplot
        lower = q25 - (box_length * 1.5)
        upper = q75 + (box_length * 1.5)

        # calculate the histogram of the series and compare values to
        # a sample normal distribution
        data = np.random.normal(loc=series.mean(), scale=1, size=100)
        _, bin_edges = np.histogram(series, bins=50, density=True)
        kernel_density = stats.gaussian_kde(data)

        # based on the previous math, calculate the lower and upper boundaries
        # to remove outliers. This should make sure to only delete outliers and
        # not delete based on quantiles.
        lower_cut_off = kernel_density.integrate_box_1d(bin_edges[0], upper)
        upper_cut_off = kernel_density.integrate_box_1d(lower, bin_edges[-1])
        lower = lower - lower_cut_off
        upper = upper + upper_cut_off

        # find indicies to remove
        indicies += list(np.argwhere(series < lower).flatten())
        indicies += list(np.argwhere(series > upper).flatten())
    
    # remove from the dataset, reset the index first to also remove "ts" values
    df.reset_index(inplace=True)
    df.drop(index=list(set(indicies)), inplace=True)
    df.drop(columns=["index"], inplace=True, errors="ignore")

    return df


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
        exclude_keys (list): A list of keys to exclude from normalization.
    """
    METHODS = {m.value: m for m in ScalerMethod}

    def __init__(self, method: str = "minmax", exclude_keys: List[str] = ["ts"], remove_outliers: bool = False) -> None:
        assert method in self.METHODS.keys(), f"Scaling method {method} is not supported."
        self._method = self.METHODS[method]
        self._exclude = exclude_keys
        self._remove = remove_outliers

        self._files = {}
        logger.debug("Fitting normalizer on dataset, set remove outliers to %s", remove_outliers)

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {NormalizeTransform.__name__: CallableDependency.IMMEDIATELY_AFTER}

    def _get_scaler(self, context: FileWorkerContext) -> FluxScaler:
        # get minimum and maximum values from previous files
        scaler = context.get(ContextParam.SCALER)
        if scaler is None:
            scaler = FluxScaler(self._method, self._exclude)

        return scaler

    def run(self, file_path: str, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        # Load the file into a pandas DataFrame, sort by column names to ensure that
        # dataframe is consistent
        df = pd.read_hdf(file_path, key="data")
        df = df.reindex(sorted(df.columns), axis=1) # type: ignore

        with mutex:
            # find and remove outliers
            if self._remove:
                df = remove_outlier(df)
                df.to_hdf(file_path, key="clean")

                day = file_path.rsplit(os.path.sep, maxsplit=1)[-1].split("_", maxsplit=1)[1][:-3]
                self._files[day] = len(df)

            # get minimum and maximum values from previous files
            scaler = self._get_scaler(context)
            scaler.fit(df)

            context.put(ContextParam.SCALER, scaler)
        return None, None

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        logger.debug("Finished fitting the normalizer. Used method %s", self._method)
        for file_dict in metadata[Meta.FILES]:
            day = file_dict[Meta.DAY.name]
            changes = self._files.get(day, None)

            if changes:
                file_dict["recordings"] = changes

        # if outliers were removed, make sure to update the metadata's info.json file
        if len(self._files) > 0:
            metadata.drop(Meta.TOTAL_RECORDINGS)
            total = sum([d["recordings"] for d in metadata[Meta.FILES]])
            metadata.put(total, Meta.TOTAL_RECORDINGS)
            logger.debug("Removing outliers resulted in a new dataset length of %s samples", total)


class NormalizeTransform(FileWorkerCallable):
    """
    Represents a transformation operation for normalization. This class is responsible
    for transforming data into the normalized form.
    
    It assumes that the `NormalizeFit` class has been run beforehand, and the
    `FileWorkerContext` contains values needed to normalize a dataframe.

    Args:
        from_other (str): Another dataset to use the normalizer from.
    """
    def __init__(self, from_other: str = "", custom: List[str] = []) -> None:
        self._scaler = None
        self._columns = list()
        if from_other:
            m = Metadata()
            m.load(from_other)

            norm_meta = m[Meta.NORMALIZER]
            path = norm_meta[Meta.NORM_PATH.name]
            self._columns = norm_meta[Meta.NORM_KEYS.name]

            with open(path, "rb") as f:
                self._scaler = pickle.load(f)
                logger.debug("Loaded normalization of other dataset.")
        
        self._custom = [[e.strip() for e in c.split(",")] for c in custom] if custom else []

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {NormalizeFit.__name__: CallableDependency.IMMEDIATELY_BEFORE}

    def _get_scaler(self, context: FileWorkerContext):
        if self._scaler:
            return self._scaler

        scaler: FluxScaler = context.get(ContextParam.SCALER)
        logger.debug("Got scaler from 'NormalizeFit'.")

        # modify the scaler if a custom one was defined
        if self._custom:
            assert scaler.method == ScalerMethod.NORMALIZE, "Custom scalings are supported on 'MinMax' scaler only."

            for i, feature in enumerate(sorted(scaler._internal_scaler.feature_names_in_)):
                name, min_, max_ = self._custom[i]
                assert name == feature, f"Expected defined name ({name}) to match scaling feature ({feature})"
                
                scaler.overwrite_column(i, float(min_), float(max_))
            logger.debug("Modified scaler with custom values.")

        self._scaler = scaler
        return scaler

    def run(self, file_path: str, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        # read the file as pandas dataframe and sort by column names
        # to be consistent with the fitted values
        df = pd.read_hdf(file_path, key="data")
        df = df.reindex(sorted(df.columns), axis=1) # type: ignore

        if not self._columns:
            self._columns = list(df.columns.values)
        else:
            to_drop = list(set(df.columns.values) - set(self._columns))
            df = df.drop(columns=to_drop, errors="ignore")

        # get the used normalization method and the fitted values
        scaler = self._get_scaler(context)
        normed = scaler.transform(df, dtype=np.float32)

        normed.to_hdf(file_path, key="norm")
        return None, None

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        # get the fitted values and method
        scaler: FluxScaler = self._get_scaler(context)

        # clear the context parameters as they are not needed anymore
        context.clear(ContextParam.SCALER)

        # save the normalizer
        path = os.path.join(metadata[Meta.PATH], "metadata", "normalizer.pkl")
        with open(path, "wb") as f:
            pickle.dump(scaler, f)

        # save the used normalization method and fitted values to the
        # dataset's metadata
        data = {
            Meta.NORM_METHOD.name: scaler.method.value if scaler else None,
            Meta.NORM_PATH.name: path,
            Meta.NORM_KEYS.name: self._columns
        }
        metadata.put(data, Meta.NORMALIZER)
        logger.debug("Finished noramlizing the dataset.")
