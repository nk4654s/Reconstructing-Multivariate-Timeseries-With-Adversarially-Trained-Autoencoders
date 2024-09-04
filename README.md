# Comprisk

This repository contains the code for training a model, which predicts the future of a compressor. The dataset required for training can be downloaded an pre-process using the [Dataflux](https://github.com/neaxplore/xplore-ai-s3prep/tree/master) repository. Hence, this code relies on the `Dataflux` python package.

# Documentation
The documentation will be generated when building the `Comprisk` package. To do so, run:
```
sh scripts/build.sh
```

## Folder Structure
When starting a training, the folder `lightning_logs/` contains the model's weights, configuration used, tensorboard, and insights into the model's performance.

### Source Code
- Comprisk/
  - cli/
  _Command line interface configuring the training and model loading options._
  - data/
  _Loading the dataset from `Dataflux` and preparing the classes for efficient training._
  - helper/
  _Contains helper methods._
  - model/
  _Contains base classes and the AI architectures._

### configs/
The `config` folder contains sample configurations, which might not lead to the best model possible. This folder is not maintained an updated all the time, as changes will occur frequently.

### docs/
Here, the generated documentation of this code will be located. The generation is based on the `sphinx` formatted python documentation.

### scripts/
Scripts to clean, start a training, and build the package are located here.

## Installation
First make sure to have the `Dataflux` code next to this repository and named `Dataflux`. To install the package, run
```
sh scripts/build.sh <cpu | cuda>
```

or build the package manuall by typing

```
conda env create -f requirements.yml
```

>**Note:** If CUDA is required, then append the `-cuda.yml` postfix to the manual installation command.

## Monitoring
The conda environment installs `tensorboard` alognside other packages. Using tensorboard, the current and past trainings can be reviewed. Simply call

```
tensorboard --logdir=lightning_logs
```

>**Note:** Make sure to run a training first, or have the `lightning_logs` folder present.
