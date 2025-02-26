import argparse
import os.path
import shutil

import onnx

from Comprisk.helper.structure import  search_models


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
    dst = os.path.join(dst.rsplit(os.path.sep, maxsplit=2)[0], "model.onnx")

    model = onnx.load(src)
    print("Loaded selected model")
    print("\tVersion:", model.model_version)
    print("\tMachine:", model.domain)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Useful tools for development and deployment.")
    parser.add_argument("--deploy", "-d", action="store_true", default=False)
    parser.add_argument("--edit", "-e", action="store_true", default=False)
    args = parser.parse_args()

    if args.deploy:
        deploy_model()

    if args.edit:
        delete_model()
