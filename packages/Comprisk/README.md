# Comprisk

This repository contains the code to train a model, which either evaluates the current state of a compressor or predicts 
its future. The dataset required for training can be downloaded and pre-processed using the 
[Dataflux](https://github.com/neaxplore/xplore-ai-cas/tree/master/packages/Dataflux) package. This will be handled 
automatically depending on the used entrypoint (e.g. `comprisk-train` or `comprisk-analyze`) and the configuration 
provided using the `-c` argument.

## Folder Structure
When starting a training, the folder `models/` contains the model's weights, configuration used, tensorboard, and 
insights into the model's performance. This folder will be located either in `dev` or `production` depending on
the configuration made before training.

### Source Code
- `Comprisk/`
  - `anomaly/`
  _The score transformers transferring the raw loss values into a number between 0 and 1._
  - `cli/`
  _Command line interface configuring the training and model loading options._
  - `data/`
  _Loading the dataset from `Dataflux` and preparing the classes for efficient training._
  - `entrypoints/`
  _This folder contains all entrypoints for the training suite. The `pipeline.py` is the most important entry point. To learn more about this one, check out the `Usage` section._
  - `helper/`
  _Contains helper methods._
  - `model/`
  _Contains base classes and the AI architectures._
  - `tune`
  _This subpackage contains code to automatically tune hyperparameters of the model. The code bases on `Optuna`._

## Monitoring
The conda environment installs `tensorboard` alongside other packages. Using tensorboard, the current and past trainings can be reviewed. Simply call

```
tensorboard --logdir=models
```

>**Note:** Make sure to run a training first, or have the `models` folder present.
