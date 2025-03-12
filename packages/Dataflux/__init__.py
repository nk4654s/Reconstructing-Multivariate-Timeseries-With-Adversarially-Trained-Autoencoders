"""The Dataflux package contains entrypoints to download and process a dataset made
out of `TSV-files` stored on an Amazon S3 server. The pre-processing steps
can be found in the entrypoint file `pipeline.py`.

A processing module has to be registered in the `modules` list at
the very top, if it should be executed as a default every time. To add new modules,
just for a single dataset configuration, define the module's class path in the yaml-config.
To create custom modules, head to the `Dataflux.modules` create a `FileWorker` subclass.

### Config Structure
To download and process a dataset, run the command `flux-pipeline -c <path-to-config>`,
whereas the config should look something like:

```
folder: dataset-name
machine-uuid: <UUID>
intervals:
  - <start> <end>
  - ...

modules:
  - class_path: <class-path-with-package>
    init_args:
      <arg-name>: <arg-value>
```

### Usage
Once a dataset has been downloaded, the subpackage provides a `pytorch` submodule to load a
dataset for machine learning. Install the Dataflux package as `pip`-package and import it.
"""
import logging
import os

from Dataflux.logger import get_logger

# to enable the debug mode, the content of CAS_DEBUG_MODE does not matter, only if it exists
DEBUG_MODE = os.environ.get("CAS_DEBUG_MODE", "0") != "0"

# switch logging mode depending on DEBUG_MODE
logging_mode = logging.DEBUG if DEBUG_MODE else logging.INFO
logger = get_logger(__name__, path="./logfile", level=logging_mode)
