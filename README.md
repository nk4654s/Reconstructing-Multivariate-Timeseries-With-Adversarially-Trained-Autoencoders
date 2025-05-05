# Compressor AI Suite 🤖📈
[![CAS CI/CD](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/pipeline.yml/badge.svg)](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/pipeline.yml)
[![Dataflux](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/dataflux-tests.yml/badge.svg)](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/dataflux-tests.yml)
[![Comprisk](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/comprisk-tests.yml/badge.svg)](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/comprisk-tests.yml)
[![Toking Tests](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/toking-tests.yml/badge.svg)](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/toking-tests.yml)

The Compressor AI Suite (CAS) mainly consists of two packages `Comprisk` and `Dataflux` located under 
`packages/`. The `Dataflux` package manages the download and processing of a compressor's dataset. Click 
[here](https://github.com/neaxplore/xplore-ai-cas/tree/master/packages/Dataflux) to learn more 
about the standalone features and development of `Dataflux`. The `Comprisk` package is responsible for training an AI based 
on the downloaded dataset. A set of multiple configuration files manages the training hyperparameters and 
specifies the exact training. More information about Comprisk and its configuration can be in the `Usage`-Section.

## Installation 📦
CAS relies on [Python](https://www.python.org), [uv](https://docs.astral.sh/uv/getting-started/installation/), and supports a variety of deep learning acceleration backends
(e.g. `CUDA`, `MPS`, ...). The installation is simplified by multiple script files, but Python must be 
installed manually.

### Development Environment
Use the `setup.sh` script to set up the development environment, which comes with an installation of `uv`. After the 
installation of `uv`, the project's dependencies will be installed. Make sure to run the following command within the 
downloaded repository. The installation may take a while.

```commandline
sh scripts/setup.sh
```

> Note: `CPU` includes the framework `MPS` to accelerate training on Apple Silicon

### Environment Variables
CAS utilizes environment variables to configure the development environment. If the `setup.sh` script of the previous
section was used to install everything, then all `ENV` vars should be configured correctly. Currently, the following
variables are set:

- `CAS_DEBUG_MODE=1` enables logging, set to `0` to disable logging. Generated AI scores will be uploaded to the test 
    device, if `CAS_DEBUG_MODE=<machine-uuid>`.
- `CAS_PROJECT_ROOT=$(pwd)` set the local repository as the project root, which is required for the logger to find all 
  submodules of CAS.

### Documentation 🎓
The developer documentation can be generated as `HTML`-files after installing the developer environment using the 
following command:

```commandline
sh scripts/docs.sh
```

## Configuration 🛠️

In CAS, everything is handled through `yaml` configuration files, which provide a flexible and scalable way to define 
complex workflows. There are two types of configurations, one required for `Comprisk` and one required for `Dataflux`.

> Note: To simplify things, one can link the Dataflux-Config to the Comprisk-Config, as `Dataflux` became part of 
the `Comprsik` pipeline.

```
configs/
    machine_1/
        0_model.yaml
        1_dataset_train.yaml
        2_dataset_validate.yaml
        3_normalization.yaml
        4_testset_1.yaml         (optional)
        ...
    machine_2/
        ...
```

The key insight here is that each model has its own dedicated pipeline configuration, which defines the specific 
training and data workflow for that model.

### Defining a Flux Configuration

A `Dataflux` or `Flux` configuration is a critical component of the system, responsible for defining the data to be 
downloaded and processed from each machine given a specific timespan. This pipeline serves as the foundation for 
producing the data used to train and validate AI models.

The dataset pipeline config requires three keys: `folder`, `machine_uuid`, and `intervals`. These keys define the dataset's
output folder, the machine's UUID to download data from, and the timespan to download from. Optional arguments are
`num_workers`, `overwrite`, and `modules`.

```yaml
folder: <path-to-dataset-folder>
machine_uuid: <machine-uuid>
intervals:
  - <start-date> <end-date>
  - ...
num_workers: <number-of-threads>
overwrite: <overwrite-existing-folder>

modules:
  ...
```

> Note: It is possible to precisely define a timespan given hours and minutes of a day. E.g. `2025-01-25T00:00:00Z` as
> start time and `2025-01-25T10:30:00Z` as end time.

#### Modules
One can configure many modules and add them to the Flux-Configuration. The general syntax is as follows:

```yaml
...
modules:
  - class_path: <package-path-to-class>
    init_args:
      <arg-name>: <arg-value>
      ...
  - ...
```

To implement a custom module, follow the implementation of `Dataflux.modules.features.Features`. This module ensures
that the dataset is _always_ consistent and includes a given list of features. Most used modules are:

- `Dataflux.modules.State`
- `Dataflux.modules.features.Features`

### Defining a Training

A training configuration defines a `trainer`, the `model`, and the `data` to load. The `trainer` describes the training 
process, including callbacks, the model's precision, maximum number epochs to train, and logging. The `model` keyword 
points to the AI architecture including hyperparameters and potential sub-modules. Finally, the `data` defines the
dataset's path, sample sequence length, and more.

> Note: The configuration allows a nested configuration to a Flux-Configuration as a dataset path. This will automatically
> check for the dataset at its desired location and download it if required.

```yaml
# CAS specific
tag: <machine-uuid>
normalizer: <path-to-normalizer-config>

trainer:
    callbacks:
        - class_path: lightning.pytorch.callbacks.EarlyStopping
          init_args:
              monitor: validate/MSELoss
              min_delta: 1.0e-05
              patience: 3
        - class_path: Comprisk.helper.checkpoint.BasicCheckpointCallback
    precision: 32
    max_epochs: <max-number-of-epochs>
data:
    train: <path-to-train-dataset-or-flux-config>
    validate: <path-to-validate-dataset-or-flux-config>
    input_sequence: <length-of-model-input-sample>
    output_sequence: <length-of-model-output-sample>
    behaviour: <"autoencoder" | "future">
    skip: <sample-resolution>
    ...
model:
    ...
```

> Note: Before training, the Flux-Configurations will be copied to the model's output folder as `dataset-train.yaml` and 
`dataset-validate.yaml`.

#### Model Configuration

The model configuration heavily depends on the `Comprisk.model.BaseModel` initialization arguments. Additional arguments
are likely, depending on the model class provided. E.g.

```yaml
model:
  class_path: <model-package-path>
  init_args:
    # BaseModel specific
    activation: tanh
    loss: MSELoss
    optimizer: Adam
    lr: 0.0001
    lr_decay: 0.9
    dropout_prob: 0.3
    # Model specific
    ...
```

### Define Hyperparameter-Tuning

The system implements a straightforward way to perform a hyperparameter search only on parameters of interest. To do so, a `yaml`  configuration can be created, which is totally optional to create a full pipeline. This config contains the specifications for the **model** hyperparameters, **data** configuration, and **rules** which may are important depending on the model architecture. 

For instance, the auto-encoder architecture requires an output dimension of the encoder which equals the decoders input dimension. A rule whould simply define the hyperparameter path of the decoder input to be the encoder output. Then, the encoder output can be part of the hyperparameter search without resulting conflicts.

More complex rules with mathematical operators such as plus, minus, multiply, and divide are possible.

```yaml
model:
    model_parameter_1:
        type: <type-int-or-float>
        min: <min-allowed-value>
        max: <max-allowed-value>
data:
    skip:
        type: int
        min: <min-allowed-value>
        max: <max-allowed-value>
rules:
    - model_parameter_2: model_parameter_1 (<equation>)
    - ...
```

## Usage 🚀
### Training
The AI training is handled Comprisk and mainly performed using the `comprisk-train` command, which uses a `YAML` 
configuration file to define the training pipeline. After installing the developer environment and building the 
documentation head to `docs/Comprisk/index.html`. Here, the configuration's structure is explained in more detail. 
Depending on the environment, the training configuration file should be located in the `configs` folder next to the 
`dataflux` files. Then, use the following command to start a training.

```commandline
comprisk-train -c path_to_config
```

> Note: When starting a training, the folder `environment/dev/models/` contains the model's weights, configuration 
> used, tensorboard, and insights into the model's performance.
> 
> When a model should be deployed  to production, then use the `dev_tools.py -d` command. This will copy the 
> configuration and model, while checking for overwrites. To minimise the folder size, the `analysis` subfolder and 
> the `*.tfevents.*` file are excluded from the model's output folder.

### Fine-Tuning
Use the same command as for training when fine-tuning a model, but re-use the `config.yaml` file in the model's output 
folder. This will automatically load the best checkpoint and continue training using the same settings as before. If a 
different dataset should be used, set `--data.train path_to_dataflux_config` (repeat this for `validate`). Alternatively
define another config containing the values of `data` only (name this file e.g. `update.yml`). Then use the following 
command.

```commandline
comprisk-train -c path_to_config_of_trained_model --data path_to_update.yml
```

### Evaluation
To apply a model for testing, use the `comprisk-analyze` command and provide a dataset to use given the argument 
`--data.validate path_to_dataflux_config`.

```commandline
comprisk-analyze -c path_to_config_of_trained_model --data.validate path_to_dataflux_config
```

## Deployment 💅🏽
This repository comes with a set of handy scripts packed in a single python file, which can be accessed by calling 

```commandline
python dev_tools.py -h
```

The Dev-Tools help organizing folders and copying a model from development (`-d`) to production environment. The deployment
will copy the `model.onnx` file to the `environment/production/models/` folder and other files to the production's
`configs` folder.

The models trained for production are included in git, causing a GitHub Action pipeline to
execute the `dev_tools.py`. The Dev-Tools calculates the current version number and updates the `pom.xml` accordingly. 
Following the `pom.xml` update, a new `*.jar` file is built by GitHub Actions and published to GitHub packages. This 
jar-file contains all `*.onnx` models located in `environment/production/models/`.

### File Structure

The model filenames are constructed by the machine's `UUID` the model was trained on, followed by the type of model. Here,
`support` stands for the support points model (refer to `packages/Compmesh`) and `model` means deep learning model
(refer to `packages/Comprisk`).

```
models/
    <UUID-A>_model.onnx
    <UUID-A>_support.onnx
configs/
    <UUID-A>/
        model/
            checkpoints/
                <checkpoint_a>.ckpt
                <checkpoint_b>.ckpt
                <checkpoint_c>.ckpt
            derived_from/       (optional)
            dataset_train.yaml
            dataset_val.yaml
            config.yaml
            ...
        support/
            ...
    ...
```