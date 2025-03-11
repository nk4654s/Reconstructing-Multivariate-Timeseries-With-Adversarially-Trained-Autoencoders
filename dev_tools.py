import argparse
import os.path
import shutil
from uuid import UUID
import xml

import onnx

from Comprisk.helper.structure import  search_models
from Dataflux.helper import get_sensors


def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        print(f"Invalid input '{val}'.")
        exit(1)


def deploy_model():
    models = search_models("environment/dev/models")
    for i, path in enumerate(models):
        print(f"({i}) {path}")

    print("Which model should be moved to deployment?")
    i = safe_cast(input(f"0-{len(models)-1}> "), int)
    src = models[i]
    dst = src.replace("environment/dev/models", "environment/production/models")
    path, _, _, version, _ = dst.rsplit(os.path.sep, maxsplit=4)

    model = onnx.load(src)
    print("Loaded selected model")
    print("\tVersion:", model.model_version)
    print("\tMachine:", model.domain)

    dst = os.path.join(path, f"{model.domain}_{model.model_version}_{version.split('_')[-1]}.onnx")
    print("\tCopy to:", dst)
    print("\tExists:", os.path.exists(dst))

    if os.path.exists(dst):
        print("Found existing model in location")
        model = onnx.load(dst)
        print("\tVersion:", model.model_version)
        print("\tMachine:", model.domain)
        answer = input("Continue [yes|no]> ")
        if answer != "yes":
            print("Abort.")
            exit(0)

    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    shutil.copyfile(src, dst)

    # copy the configs
    dst = os.path.join("environment", "production", "configs", model.domain)
    if not os.path.exists(dst):
        os.makedirs(dst)

    src = os.path.dirname(src)
    config = os.path.join(src, "config.yaml")
    shutil.copyfile(config, os.path.join(dst, f"config_{model.model_version}.yaml"))

    dataset = os.path.join(src, "dataset.yaml")
    shutil.copyfile(dataset, os.path.join(dst, f"dataset_{model.model_version}.yaml"))
    print("Done.")


def update_pom():
    # get the total model version (simply by calculating the model version sum)
    models = [f for f in os.listdir("environment/production/models") if "onnx" in f]
    models = [os.path.join("environment/production/models", f) for f in models]
    model_versions = sum([onnx.load(f).model_version for f in models])

    # get the number of model domains
    model_count = len(set([onnx.load(f).domain for f in models]))

    # get the code version
    code = 0
    for package in os.listdir("packages/"):
        if package.startswith("."):
            continue

        with open(os.path.join("packages", package, package, "version.txt"), "r") as f:
            major, minor = f.read().split(".")
            code += safe_cast(major, int, 1)
            model_versions += safe_cast(minor, int, 0)

    # modify the pom.xml version tag
    tag = "{http://maven.apache.org/POM/4.0.0}"
    xml.etree.ElementTree.register_namespace("", "http://maven.apache.org/POM/4.0.0")
    et = xml.etree.ElementTree.parse("pom.xml")
    root = et.getroot()

    version_child = root.find(tag + "version")
    version_child.text = f"{code}.{model_count}.{model_versions}"

    # Write back to file
    et.write("pom.xml")
    print("Done.")


def delete_model():
    models = search_models("environment/production/models")
    for i, path in enumerate(models):
        print(f"({i}) {path}")

    print("Which model should be deleted?")
    i = safe_cast(input(f"0-{len(models)-1}> "), int)
    path = models[i]

    model = onnx.load(path)
    print("Loaded selected model")
    print("\tVersion:", model.model_version)
    print("\tMachine:", model.domain)

    answer = input("Continue [yes|no]> ")
    if answer != "yes":
        print("Abort.")
        return None

    os.remove(path)
    print("Done.")


def show_sensors(uuid_str):
    rename = get_sensors(UUID(uuid_str))
    result = []
    for i, (uuid, name) in enumerate(rename.items()):
        result.append((uuid, name))

    for uuid, name in sorted(result, key=lambda x: x[1]):
        print(f"- {uuid}\t# {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Useful tools for development and deployment.")
    parser.add_argument("--deploy", "-d", action="store_true", default=False)
    parser.add_argument("--edit", "-e", action="store_true", default=False)
    parser.add_argument("--sensors", "-s", type=str, default="")
    args = parser.parse_args()

    if args.edit:
        delete_model()

    if args.deploy:
        deploy_model()

    if args.sensors:
        show_sensors(args.sensors)

    update_pom()
