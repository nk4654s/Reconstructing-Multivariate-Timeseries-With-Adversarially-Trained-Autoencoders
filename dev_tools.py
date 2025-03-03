import argparse
import os.path
import shutil
from uuid import UUID

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
        exit(0)

    os.remove(path)
    print("Done.")


def show_sensors(uuid_str):
    rename = get_sensors(UUID(uuid_str))
    result = []
    for i, (uuid, name) in enumerate(rename.items()):
        result.append((uuid, name))

    for uuid, name in sorted(result, key=lambda x: x[1]):
        print(f"- {uuid}\t# {name}")


def build_jar():
    # temp copy files
    src = "environment/production/models"
    dst = "models"
    shutil.copytree(src, dst)

    # move models to subfolder
    sub = "neax/ai/resources/"
    os.makedirs(os.path.join(dst, sub))
    for m in os.listdir(dst):
        if not ".onnx" in m:
            continue

        shutil.move(os.path.join(dst, m), os.path.join(dst, sub, m))

    # zip and rename the jar
    shutil.make_archive(dst, "zip", dst)
    os.rename(f"{dst}.zip", f"{dst}.jar")

    # remove the dir again
    shutil.rmtree(dst)
    print("Jar built. Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Useful tools for development and deployment.")
    parser.add_argument("--deploy", "-d", action="store_true", default=False)
    parser.add_argument("--build", "-b", action="store_true", default=False)
    parser.add_argument("--edit", "-e", action="store_true", default=False)
    parser.add_argument("--get-sensors", "-s", type=str, default=None)
    args = parser.parse_args()

    if args.edit:
        delete_model()

    if args.deploy:
        deploy_model()

    if args.build:
        build_jar()
        exit(0)

    if args.sensors:
        show_sensors(args.sensors)
