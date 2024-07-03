# Installation
Make sure to have [Miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install) installed on your system and this repository cloned. Then, open a terminal in the root folder of this repository and run:

```
conda env create -f environment.yml
```

Wait for the environment to be set up. Then, activate it with:

```
conda activate s3prep
```

# Usage
The main script is `main.py` which can be configured using arguments. To see the full list of available options run:

```
python main.py -h
```

For example, to download the sensor data of the default machine of November 2020 and store a report alongside the dataset run:

```
python main.py --folder dataset/test.h5 --interval 2020-11-01 2020-11-30 --report-folder dataset/report
```