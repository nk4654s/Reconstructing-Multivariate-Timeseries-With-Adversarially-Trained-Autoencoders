"""This submodule contains the downloading and pre-/postprocessing pipeline.
The pipeline is built generic, as it can be configured by defining
pre-/postprocessing modules. Example code can be found in the entrypoints
`Dataflux.preprocess` and `Dataflux.postprocess`."""
from Dataflux.data.s3io import S3, BucketType
from Dataflux.data.dfio import DFIO
