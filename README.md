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