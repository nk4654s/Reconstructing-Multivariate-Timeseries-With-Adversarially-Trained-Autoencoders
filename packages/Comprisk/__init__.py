"""
## Overview
This package includes code to train a model that can either evaluate the current state of a compressor or predict its
future behavior based on given data. The necessary dataset for training this model is available and must be
preprocessed using the Dataflux package, which you can find [here](Dataflux.html).

Each submodule of the package contributes to different parts of the overall package. Model architectures are defined
in the `model` section, hyperparameter tuning takes place in the `tune` folder, and anomaly detection calculations are
handled within the `anomaly` module. The installed package comes with multiple command line interfaces which work
together in the `comprisk-pipeline`.

## Configuration
The following `yaml` contains a template a configuration file for running the `comprisk-train` entrypoint, which is
used to train models on compressor data. The configuration defines different steps in the pipeline such as loading the
dataset (from `data` section), training and validating the model (`model`, `trainer` and `score_transformer` sections
respectively).

```yaml
environment: [dev | production]
tag: [str | machine's UUID]
normalizer: [path to normalize.yaml]

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

import Comprisk.data as data
import Comprisk.helper as helper
import Comprisk.model as model
import Comprisk.cli as cli
import Comprisk.tune as tune
import Comprisk.anomaly as anomaly
import Comprisk.entrypoints as entrypoints

__all__ = ["DEBUG_MODE", "anomaly", "cli", "data", "entrypoints", "model", "helper", "tune"]
