# Installation
Make sure to have [Miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install) installed on your system and this repository cloned. Then, open a terminal in the root folder of this repository and run:

```
conda env create -f environment.yml
```

Wait for the environment to be set up. Then, activate it with:

```
conda activate dataflux
```

To build and link the library, run:

```
pip install -e .
```

> Note: The last output line should contain `Successfully installed Dataflux-1.0`.

# Usage
The application main script can be executed by typing `flux-pipeline`. In general, the `flux-pipeline` downloads a dataset from the Amazon S3 bucket, runs some preprocessing and stores the files on the local disk. Afterward, the data is further processed (e.g. normalize the dataset). The recommended way to execute the pipeline is by configuring a `yaml`-config file. An example for a config file is available under `configs/green_config.yaml`.

# Modules
## Preprocessing
The preprocessing module's code is located under `Dataflux/modules/preprocess`. Adding a new module can be achieved, by simply inheriting the base class from `Dataflux/modules/preprocess/module.py` and publishing the class to the package in `Dataflux/modules/preprocess/__init__.py`. Finally, define the created class path in a config file to enable the newly created module.

Available modules are:
- Homogeneous columns and rows
- Interpolate values (e.g. to fill NaN)
- Split the dataset when machine goes offline
- Rename the columns from UUID to human-readable name
- Create a report to analyze the dataset

## Postprocessing
Similar to preprocessing, the code is located under `Dataflux/modules/postprocess` but a new module is added to `Dataflux/postprocess.py` instead. Also, the postprocess modules are build to run on multiple threads enabling faster postprocessing.

Available modules are:
- Normalization (MinMax, Standardize)
- Detect and remove outlier
- Remove one ore multiple columns by name
- Add historical weather data from an external source