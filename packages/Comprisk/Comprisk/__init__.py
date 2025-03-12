"""
## Overview
This package includes code to train a model that can either evaluate the current state of a compressor or predict its
future behavior based on given data. The necessary dataset for training this model is available and must be
preprocessed using the Dataflux package, which you can find
[here](https://github.com/neaxplore/xplore-ai-dataflux/tree/master). The `Dataflux` package is a requirement
for running this software code.

Each submodule of the software contributes to different parts of the overall package. Model architectures are defined
in the `model` section, hyperparameter tuning takes place in the `tune` folder, and anomaly detection calculations are
handled within the `anomaly` module. The installed package comes with multiple command line interfaces which work
together in the `comprisk-pipeline`. The `cli` submodule in Dataflux is responsible for managing any configurations
needed when starting up the program.

## Configuration
The following `yaml` contains a template a configuration file for running the `comprisk-train` software, which is
used to train models on compressor data. The configuration defines different steps in the pipeline such as loading the
dataset (from `data` section), training and validating the model (`model`, `trainer` and `score_transformer` sections
respectively).

```yaml
environment: [dev | production]
tag: [str | machine's UUID]
upload_config: [path to dataflux pipeline]

score_transformer:
  mode: [check out Comprisk.anomaly.variants]

model:
  class_path: [package path to model]
  init_args:
    ...

trainer:
  accelerator: [cpu | cuda | mps | auto]
  max_epochs: [number of training epochs]
  callbacks:
    ...

data:
  train: [path to dataflux pipeline | dataset folder]
  validate: [path to dataflux pipeline | dataset folder]
  ...
```
"""

from Dataflux import DEBUG_MODE

import Comprisk.anomaly as anomaly
import Comprisk.cli as cli
import Comprisk.data as data
import Comprisk.entrypoints as entrypoints
import Comprisk.helper as helper
import Comprisk.model as model
import Comprisk.tune as tune

__all__ = ["DEBUG_MODE", "anomaly", "cli", "data", "entrypoints", "model", "helper", "tune"]
