# Comprisk

This repository contains the code to train a model, which either evaluates the current state of a compressor or predicts its future. The dataset required for training can be downloaded and pre-processed using the [Dataflux](https://github.com/neaxplore/xplore-ai-dataflux/tree/master) repository. The `Dataflux`-package is one of the requirements for this code to run.

## Installation
The documentation will be generated when building the `Comprisk` package. Building the repository requires `Dataflux` to be located next to this code. Check out the installation of `Dataflux` [here](https://github.com/neaxplore/xplore-ai-dataflux). After all prerequisite were accomplished, continue with building `Comprisk`.
```
sh scripts/build.sh [cpu|cuda (default=cpu)] [username (default=$USER)]
```

## Documentation
The documentation will be generated when building the package. Follow the instructions provided by the `Installation` section of the Readme. Afterwards, the documentation should be located in a folder called `docs/`.

## Folder Structure
When starting a training, the folder `lightning_logs/` contains the model's weights, configuration used, tensorboard, and insights into the model's performance.

### Source Code
- `Comprisk/`
  - `analyzer/`
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

### Misc
#### configs/
The `config` folder contains sample configurations, which might not lead to the best model possible. This folder is not maintained an updated all the time, as changes will occur frequently.

#### scripts/
Scripts to clean, start a training, and build the package are located here.

#### docs/ (Generated)
Here, the generated documentation of this code will be located. The generation is based on the `sphinx` formatted python documentation.

## Usage (Locally)
As this repository is under active development, the entrypoints and their configuration may change faster than this Readme. A good starting point after installation is to check the `setup.py` content at the end of file.

```python
...
    keywords="Compressors,AI,Amazon S3,S3",
    name="Comprisk",
    entry_points={
        "console_scripts": [
            "comprisk-analyzer=Comprisk.entrypoints.analyzer:cli_main",
            "comprisk-generator=Comprisk.entrypoints.generate:cli_main",
            "comprisk-pipeline=Comprisk.entrypoints.pipeline:cli_main",
            "comprisk-train-score=Comprisk.entrypoints.scoring:cli_main",
            "comprisk-trainer=Comprisk.entrypoints.trainer:cli_main",
            "comprisk-tuner=Comprisk.entrypoints.tuner:cli_main",
        ]
    },
    packages=find_packages(include=[
        "Comprisk", "Comprisk.*"
    ]),
...
```

To find more help about one entrypoint, then ask the terminal for help (`-h`). E.g.
```
comprisk-pipeline -h
```

Most entrypoints can be configured in advance using `yaml`-configuration files. Examples of such files are located inside the `config/` folder and named `pipeline.yaml`. A pipeline can define 6 actions `dataset`, `tuner`, `trainer`, `score_trainer`, `generator`, and `analyzer`. 

### Dataset
The dataset key enables the `Dataflux` pipeline command for the `Comprisk` pipeline. At the start, `Comprisk` will 
download defined datasets and pre-process them. To define a dataset to download, provide a path to a `Dataflux` 
pipeline `yaml`-configuration. A configuration could look like

```yaml
dataset:
  - config: path/to/dataset/config_a.yaml
  - config: path/to/dataset/config_b.yaml
  - config: path/to/dataset/config_c.yaml
    copy_to_output: desired_name.yaml
  - ...
```

Using the flag `copy_to_output`, the dataset will not be downloaded but the configuration will be copied to the
created model output folder. This comes in handy, when deploying a model within a docker container.

### Tuner
TBD

### Trainer
Similar to the `dataset` key, the `trainer` will simply call an entrypoint given another `yaml`-configuration. This time, the pipeline will execute the `comprisk-trainer` entrypoint and provide the configuration. Moreover, one can define the trainings `tag` in form of a string to manage multipe runs.

```yaml
trainer:
  - config: path/to/dataset/config_a.yaml
    tag: run_01

  - config: path/to/dataset/config_a.yaml
    tag: run_02
    data.path: differen/path/than/before
```

> Note: To overwrite parameters from the pipeline config, write the full parameters key and define a new value to it. Nested parameters can be accessed by separating the keys using a `.`.

If mutliple training configurations were set, then the pipeline will execute each training in the specified order. The output model of a preceeding model will be used as a baseline and the training will continue giving the new configuration.

> Note: Follow-up trainings may raise errors as configurations are very sensitive. Be caution and triple check advanced configurations.

### Score Trainer
This entrypoint trains the score transformer for a specific model on the validation set. Next to an appropriate scoring method, it is recommended to overwrite the dataset's stride value. For faster calculation, a stride value equals the input/output size of the model should be set.

```yaml
score_trainer:
  score.method: linear
  data.stride: 60 # or whatever the model uses
```

### Generator
The generator will download and preprocess a given dataset, then load an already trained model and execute it on the downloaded dataset. Then, the results are passed through the score transformer and uploaded to Xplore. One requirement of this entrypoint to work is an existing score transformer in the model's `metadata/` folder. If the mathcing `pkl`-file does not exist, run the `comprisk-train-score` entrypoint beforehand.

```yaml
generate:
  config: path/to/dataset/pipeline/config.yaml
  data.stride: 60 # or whatever the model uses
```

### Analyzer
The analyzer is a great tool for debugging and fast model evaluations on real data. The usage of this entrypoint matches the pattern of all previous entrypoints. As this one does not train a model but loads one, provide the path to a saved `yaml`-configuration in the `models/` output folder. Such a path could look like `models/<Architecture>/<Tag>/version_<v_num>/config.yaml`.

```yaml
analyzer:
  - dataset: path/to/dataset/folder
    prefix: folder/subfolder

  - dataset: path/to/dataset/folder
    prefix: folder/subfolder
```

### Summary
Call the follwing command

```
comprisk-pipeline -c pipeline.yaml
```

whereas the pipeline looks something like this

```yaml
dataset:
  - config: ./datasets/trainset.yaml
  - config: ./datasets/valset.yaml
  - config: ./datasets/testset.yaml

# tuner:
#   TBD

trainer:
  - config: ./training/pre-train.yaml
    data.stride: 30
    tag: initial

  - config: ./training/fine-tune.yaml
    data.stride: 1
    tag: final

score_trainer:
  score.method: linear
  data.path: valset/
  data.stride: 60 # or whatever the model uses

generate:
  config: ./dataset/upload.yaml
  data.stride: 60 # or whatever the model uses

analyzer:
  - dataset: valset/
    prefix: analysis/valset

  - dataset: testset/
    prefix: analysis/testset
```

## Usage (Docker)
TBD

## Monitoring
The conda environment installs `tensorboard` alongside other packages. Using tensorboard, the current and past trainings can be reviewed. Simply call

```
tensorboard --logdir=models
```

>**Note:** Make sure to run a training first, or have the `models` folder present.
