"""
## Overview
This package includes code to train a model that can either evaluate the current state of a compressor or predict its
future behavior based on given data. The necessary dataset for training this model is through the Dataflux package,
which you can find [here](Dataflux.html).

## Configuration
The following `yaml` contains a template a configuration file for running the `cas-train` entrypoint, which is
used to train models on compressor data. The configuration defines different steps in the pipeline such as loading the
dataset (from `data` section), training and validating the model (`model` and `trainer` sections).

```yaml
environment: dev
tag: [str | machine's UUID]

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
