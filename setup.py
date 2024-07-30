#!/usr/bin/env python
import os
from setuptools import setup, find_packages

# get package version
VERSION = ""
with open(os.path.join("Comprisk", "version.txt"), "r", encoding="UTF-8") as file:
    VERSION = file.read().strip()

# get readme content
README = ""
with open("README", "r", encoding="UTF-8") as file:
    README = file.read()

# get requirements from conda environment
os.system("pip freeze > requirements.txt")

requirements = []
with open("requirements.txt", "r", encoding="UTF-8") as f:
    for line in f.readlines():
        if "==" in line:
            requirements.append(line.split("==")[0])
        elif "@" in line:
            requirements.append(line.split(" @")[0].strip())

with open("requirements.txt", "w", encoding="UTF-8") as f:
    for line in requirements:
        f.write(line + "\n")

# run the setup
setup(
    author="Jonas Leuchtenberger",
    python_requires=">=3.9",
    classifiers=[],
    description="Dataset download and pre-processing application.",
    install_requirements=requirements,
    keywords="Dataset,Pre-Processing,Amazon S3,S3",
    name="Comprisk",
    entry_points={
        "console_scripts": [
            "comprisk-train=Comprisk.train:cli_main",
            "comprisk-load=Comprisk.load:cli_main",
        ]
    },
    packages=find_packages(include=[
        "Comprisk", "Comprisk.*"
    ]),
    package_dir={"Comprisk": "Comprisk"},
    include_package_data=True,
    url="https://github.com/neaxplore/xplore-ai-comprisk",
    version=VERSION,
    zip_safe=True
)