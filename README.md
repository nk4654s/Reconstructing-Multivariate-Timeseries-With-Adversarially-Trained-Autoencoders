# Compressor AI Suite
[![Dataflux](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/dataflux-tests.yml/badge.svg)](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/dataflux-tests.yml)
[![Comprisk](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/comprisk-tests.yml/badge.svg)](https://github.com/neaxplore/xplore-ai-cas/actions/workflows/comprisk-tests.yml)

The Compressor AI Suite (CAS) mainly consists of two `PIP` packages Comprisk and Dataflux located under 
`packages/`. The Dataflux package manages the download and pre-processing of a compressor. Click 
[here](https://github.com/neaxplore/xplore-ai-cas/tree/master/packages/Dataflux) to learn more 
about the standalone features and development of Dataflux. The Comprisk package is responsible for training an AI based 
on the downloaded dataset by Dataflux. A set of multiple configuration files manages the training hyperparameters and 
specifies the exact training. More information about Comprisk and its configuration can be found 
[here](https://github.com/neaxplore/xplore-ai-cas/tree/master/packages/Comprisk). 

## Installation
CAS relies on [Python](https://www.python.org), [Miniconda](https://docs.anaconda.com/miniconda/), and 
[Docker](https://www.docker.com) and supports a variety of deep learning acceleration backends
(e.g. `CUDA`, `MPS`, ...). The installation is simplified by multiple script files, but Python and Docker must be 
installed manually.

### Development Environment
Use the `setup.sh` script to set up the development environment, which comes with an installation of Miniconda. As 
the user, pass `$USER` to install the environment for the current user and as second argument either pass `CUDA` or 
`CPU`.

```commandline
sh scripts/setup.sh $USER cpu
```

> Note: `CPU` includes the framework `MPS` to accelerate training on Apple Silicon

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

> Note: Press [here](https://github.com/neaxplore/xplore-ai-cas/tree/master/environment/production/models/README.md) 
> to learn more about the production folder structure.

If you would like to have a command to copy and paste into the CLI, then use the template:

```commandline
docker build --no-cache -f dockerfiles/docker-nocuda.dockerfile -t inference .
```