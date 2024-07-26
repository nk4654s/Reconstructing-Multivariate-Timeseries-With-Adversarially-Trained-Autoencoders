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
The application consits of two main scripts `flux-pre` and `flux-post`. As the amount of processing modules and the configuration of those might change, run
```
<flux-pre | flux-post> --help
```
to get more information. In general, the `flux-pre` downloads a dataset from the Amazon S3 bucket, runs some preprocessing and stores the files on the local disk. Afterward, the `flux-post` can be used to further process the data (e.g. normalize the dataset).

# Modules
## Preprocessing
The preprocessing module's code is located under `Dataflux/modules/preprocess`. Adding a new module can be achieved, by simply inheriting the base class from `Dataflux/modules/preprocess/module.py` and publishing the class to the package in `Dataflux/modules/preprocess/__init__.py`. Finally, import the created module in `Dataflux/preprocess.py` and add the class name to `modules = []` at the start of the file.

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