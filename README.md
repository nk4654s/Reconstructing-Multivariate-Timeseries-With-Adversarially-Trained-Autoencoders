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
The AI training is handled Comprisk and mainly performed using the `comprisk-pipeline` command. To apply a model for
testing, use the `comprisk-analyzer` command. Both are explained in more detail 
[here](https://github.com/neaxplore/xplore-ai-cas/tree/master/packages/Comprisk#usage-locally).

> Note: For production usage of a model, check out the `Docker` section of this document.

## Docker
This repository uses docker to deploy all trained models located in the production environment. The model folder must
include the `config.yaml` which equals to the training configuration, a `upload.yaml` which defines the dataset to
download for inference, and model checkpoints in the `checkpoint/*.ckpt` folder. As the model's output is send to an
AWS server, no persistent volumes are required.

Usually, the docker will be build when updating the master branch while changing the `environment/production/models` 
folder. This will trigger GitHub actions and start the specified workflow immediately.

> Note: Press [here](https://github.com/neaxplore/xplore-ai-cas/tree/master/environment/production/models/README.md) 
> to learn more about the production folder structure.

If you would like to have a command to copy and paste into the CLI, then use the template:

```commandline
docker build -f dockerfiles/docker.dockerfile -t cas-uv:latest .
```
