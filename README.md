# Compressor AI Suite
[![CAS CI/CD](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/pipeline.yml/badge.svg)](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/pipeline.yml)
[![Dataflux](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/dataflux-tests.yml/badge.svg)](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/dataflux-tests.yml)
[![Comprisk](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/comprisk-tests.yml/badge.svg)](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/comprisk-tests.yml)
[![Toking Tests](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/toking-tests.yml/badge.svg)](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/toking-tests.yml)

The Compressor AI Suite (CAS) mainly consists of two `PIP` packages Comprisk and Dataflux located under 
`packages/`. The Dataflux package manages the download and pre-processing of a compressor. Click 
[here](https://github.com/neaxplore/xplore-ai-cas/tree/master/packages/Dataflux) to learn more 
about the standalone features and development of Dataflux. The Comprisk package is responsible for training an AI based 
on the downloaded dataset by Dataflux. A set of multiple configuration files manages the training hyperparameters and 
specifies the exact training. More information about Comprisk and its configuration can be found 
[here](https://github.com/neaxplore/xplore-ai-cas/tree/master/packages/Comprisk). 

## Installation
CAS relies on [Python](https://www.python.org), [uv](https://docs.astral.sh/uv/getting-started/installation/), and 
[Docker](https://www.docker.com) and supports a variety of deep learning acceleration backends
(e.g. `CUDA`, `MPS`, ...). The installation is simplified by multiple script files, but Python and Docker must be 
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

### Documentation
The developer documentation can be generated as `HTML`-files after installing the developer environment using the 
following command:

```commandline
sh scripts/docs.sh
```

## Usage
### Training
The AI training is handled Comprisk and mainly performed using the `comprisk-train` command, which uses a `YAML` 
configuration file to define the training pipeline. After installing the developer environment and building the 
documentation head to `docs/Comprisk/index.html`. Here, the configuration's structure is explained in more detail. 
Depending on the environment, the training configuration file should be located in the `configs` folder next to the 
`dataflux` files. Then, use the following command to start a training.

```commandline
comprisk-train -c path_to_config
```

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

## Dev-Tools
This repository comes with a set of handy scripts which can be accessed by calling 

```commandline
python dev_tools.py -h
```

The tool helps organizing folders and copying a model from development (`-d`) to production environment.