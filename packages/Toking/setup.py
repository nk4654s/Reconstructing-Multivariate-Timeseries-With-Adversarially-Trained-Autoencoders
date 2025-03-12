#!/usr/bin/env python
import os

from setuptools import find_packages, setup

# get package version
VERSION = ""
with open(os.path.join("Toking", "version.txt"), "r", encoding="UTF-8") as file:
    VERSION = file.read().strip()

# get readme content
README = ""
with open("README.md", "r", encoding="UTF-8") as file:
    README = file.read()

# run the setup
setup(
    author="Jonas Leuchtenberger",
    python_requires=">=3.9",
    classifiers=[],
    description="This program provides access to a valid API access token.",
    install_requirements="pycognito",
    keywords="Amazon, Cognito, Token, Dataflux, Comprisk",
    name="Toking",
    entry_points={
        "console_scripts": [
            "toking=Toking.aws:main",
        ]
    },
    packages=find_packages(include=[
        "Toking", "Toking.*"
    ]),
    url="https://github.com/neaxplore/xplore-ai-cas/packages/Toking",
    version=VERSION,
    zip_safe=True
)