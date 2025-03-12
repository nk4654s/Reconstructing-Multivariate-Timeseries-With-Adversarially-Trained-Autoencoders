#!/usr/bin/env python
import os

from setuptools import find_packages, setup

# get package version
VERSION = ""
with open(os.path.join("Dataflux", "version.txt"), "r", encoding="UTF-8") as file:
    VERSION = file.read().strip()

# get readme content
README = ""
with open("README.md", "r", encoding="UTF-8") as file:
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
    name="Dataflux",
    entry_points={
        "console_scripts": [
            "flux-pipeline=Dataflux.pipeline:main",
            "flux-merge=Dataflux.merge:main",
        ]
    },
    packages=find_packages(include=[
        "Dataflux", "Dataflux.*"
    ]),
    package_dir={"Dataflux": "Dataflux"},
    include_package_data=True,
    url="https://github.com/neaxplore/xplore-ai-s3prep",
    version=VERSION,
    zip_safe=True
)