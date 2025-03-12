#!/usr/bin/env python
import os
from setuptools import setup, find_packages

# get package version
VERSION = ""
with open(os.path.join("Compmesh", "version.txt"), "r", encoding="UTF-8") as file:
    VERSION = file.read().strip()

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
    description="AI training suite to predict a compressor's health state and future sequence curve.",
    install_requirements=requirements,
    keywords="Compressors,AI,Amazon S3,S3",
    name="Compmesh",
    entry_points={
    },
    packages=find_packages(include=["Compmesh", "Compmesh.*"]),
    package_dir={"Compmesh": "Compmesh"},
    include_package_data=True,
    url="https://github.com/neaxplore/xplore-ai-comprisk",
    version=VERSION,
    zip_safe=True,
)