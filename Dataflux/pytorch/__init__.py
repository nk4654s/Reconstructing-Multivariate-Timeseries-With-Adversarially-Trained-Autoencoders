"""The `pytorch` submodule contains code to load a downloaded dataset
as `pytorch.Dataset` efficiently. This includes caching files, randomly
sampling from files, defining the dataset's metadata and more. This submodule
is mainly required, when executing machine learning on a dataset."""
from Dataflux.pytorch.dataset import Dataset
from Dataflux.pytorch.utils import collate_fn
from Dataflux.pytorch.sampler import RandomBatchSampler
from Dataflux.pytorch.metadata import Metadata
