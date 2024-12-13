"""The Dataflux package contains entrypoints to download and process a dataset made
out of `TSV-files` stored on an Amazon S3 server. The pre- and post-processing steps
can be found in the entrypoint file `pipeline.py`.

A processing module has to be registered in the `modules_pre` or `modules_post` list at
the very top, if it should be executed as a default every time. To add new modules,
just for a single dataset configuration, define the module's class path in the yaml-config.
To create custom modules, head to the `Dataflux.modules` folder and choose either the 
`postprocess` or `preprocess` submodule.

### Config Structure
To download and process a dataset, run the command `flux-pipeline -c <path-to-config>`,
where as the config should look something like:

```
folder: dataset-name
machine-uuid: <UUID>
intervals:
  - <start> <end>
  - ...

preprocessing:
  - class_path: <class-path-with-package>
    init_args:
      <arg-name>: <arg-value>

postprocessing:
  - class_path: <class-path-with-package>
    init_args:
      <arg-name>: <arg-value>
```

### Usage
Once a dataset has been downloaded, the subpackage provides a `pytorch` submodule to load a
dataset for machine learning. Install the Dataflux package as `pip`-package and import it.
"""
import logging
from Dataflux.helper.logger import get_logger

logger = get_logger(__name__, path="logfile", level=logging.DEBUG)
