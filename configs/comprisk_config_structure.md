# Comprisk Structure

## Overview

In our system, everything is handled through `yaml` configuration files, which provide a flexible and scalable way to define complex workflows.

To get started, you'll need to create several `yaml` configs that work together. These configs are organized into a hierarchical structure, with smaller, focused configs defining the download and processing of a dataset as well as training and evaluating a model. These smaller configs are then combined into a larger Pipeline Config, which defines the overall workflow for training, loading, and applying an AI model. The pipeline config specifies which model tag to train or load (e.g., "model-v1" or "model-v2"), as well as which machines' UUIDs should be associated with this particular pipeline.

```
- configs/
    - machine_1/
        - dataset-pipeline-train.yml
        - dataset-pipeline-generate.yml
        - training-config.yml
        - tune-hyperparameters.yml
        - pipeline-training.yml
        - pipeline-generate.yml
    - machine_2/
        ...
```

The key insight here is that each machine UUID has its own dedicated pipeline config, which defines the specific training and loading/data generation workflow for that machine. This ensures that each machine's data is processed and analyzed separately as each machine is configured differently.



## Defining a Dataset Pipeline

A dataset pipeline is a critical component of our system, responsible for defining the data to be downloaded and processed from each machine given a specific timespan. This pipeline serves as the foundation for generating the analysis data used to train and validate AI models.

The dataset pipeline config defines two primary stages: **preprocessing** and **postprocessing**. These stages are sequential in nature, with preprocessing tasks applied directly on individual files after they are downloaded, while postprocessing tasks iterate over the entire dataset once all files have been downloaded and preprocessed.

```yaml
folder: <dataset-name>
machine_uuid: <machine-uuid>
intervals:
    - <start-date> <end-date>
    - ...
preprocessing:
    - class_path: <path-to-module-class>
      init_args:
        <module-specific-init-args>
postprocessing: 
    - class_path: <path-to-module-class>
      init_args:
        <module-specific-init-args>
```

The configuration system requires a unique dataset pipeline config for each machine. This is because the machine's configuration and sensor data differ from one machine to another, requiring tailored preprocessing and postprocessing tasks for each individual machine.

Another important consideration is that many sensors are calculated from others, making them dependent on those other sensors. In such cases, these derived sensors should be excluded from the training process, as they would introduce unnecessary complexity and noise into the model.

To accommodate this requirement, our system incorporates a `RemoveColumns` module that can be configured to exclude specific sensors or columns from the dataset. This module is essential for ensuring that only independent and relevant data are used in the training process.

## Defining a Training

A training configuration defines a trainer, the model, and the dataset to load. The trainer describes the training process, including callbacks, the model's precision, maximum number epochs to train, and logging. The model keyword points to the AI architecture including hyperparameters and potential sub-modules. Finally, the dataset  to use is defined alongside the sequence length to train with as well as the sample resolution.

```yaml
trainer:
    callbacks:
        - class_path: <callback-class-path>
          init_args:
    precision: 32
    max_epochs: <max-number-of-epochs>
model:
    activation: <last-layer-function>
    loss: <training-metric>
    optimizer: <strategy-to-use>
    ...
data:
    train: <path-to-train-dataset>
    validate: <path-to-validate-dataset>
    skip: <sample-resolution>
    input_sequence: <length-of-model-input>
    ...
```

## Define Hyperparameter-Tuning

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

## Define a Full-Pipeline

Our system allows constructing a pipeline combining the previously described configurations. Here, one can define the datasets to download and process, the hyperparameter search, the training, evaluation, and generation of scores. The pipeline is fully customizable and does not require each keyword to be defined, for instance if the model's hyperparameters should not be fine-tuned.

Moreover, some keywords allow the definition of multiple pipeline steps, which helps if multiple dataset or training steps are necessary.

```yaml
dataset:
    - config: <path-to-dataset-config-pipeline>
    - ...
tuner:
    config: <path-to-tuner-config>
    tuner_trials: <number-of-tuning-steps>
trainer:
    - config: <first-training-config-file>
      tag: <name-of-training>
      prefer_tuned: <true-if-tuner-set>
    - ...
analyzer:
    - dataset: <path-to-dataset>
      prefix: <path-to-analysis-result>
    - ...
generate:
    model: <path-to-model-folder>
    dataset: <path-to-dataset-config-pipeline>
```


