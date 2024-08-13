"""The Dataflux package contains entrypoints to download and process a dataset made
out of `TSV-files` stored on an Amazon S3 server. The pre- and post-processing steps
can be found in the entrypoint files `preprocess.py` and `postprocess.py`.
A processing module has to be registered in the `modules` list at the very top. To add
new modules, head to the `Dataflux.modules` submodule.

### Usage
To use a downloaded dataset, the package provides a `pytorch` submodule. Install the
Dataflux package as `pip`-package and import it.
"""
