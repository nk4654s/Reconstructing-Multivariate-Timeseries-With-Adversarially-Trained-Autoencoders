import argparse
import os.path
import shutil
import xml
from uuid import UUID

import numpy as np
from onnxruntime import InferenceSession

from Comprisk.helper import ONNXArgs
from Comprisk.helper.structure import search_models, copy_folder
from Dataflux.helper import get_sensors


def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        print(f"Invalid input '{val}'.")
        exit(1)


def load_existing_onnx():
    models = search_models("environment/production/models")
    for i, path in enumerate(models):
        print(f"({i}) {path}")

    i = safe_cast(input(f"0-{len(models)-1}> "), int)
    path = models[i]

    onnx = ONNXArgs.load_from_model(path)
    print("Loaded selected model")
    print("\tProducer:", onnx.producer)
    print("\tMachine:", onnx.machine_uuid)
    print("\tVersion:", onnx.version)
    print("\tCreation:", onnx.creation_date)
    return path, onnx


def deploy_model():
    models = search_models("environment/dev/models")
    for i, path in enumerate(models):
        print(f"({i}) {path}")

    print("Which model should be moved to deployment?")
    i = safe_cast(input(f"0-{len(models)-1}> "), int)
    src = models[i]
    dst = src.replace("environment/dev/models", "environment/production/models")
    path, _, _, version, _ = dst.rsplit(os.path.sep, maxsplit=4)

    onnx = ONNXArgs.load_from_model(src)

    print("Loaded selected model")
    print("\tProducer:", onnx.producer)
    print("\tMachine:", onnx.machine_uuid)
    print("\tVersion:", onnx.version)
    print("\tCreation:", onnx.creation_date)

    dst = os.path.join(path, f"{onnx.machine_uuid}_{os.path.basename(src)}")
    print("\tCopy to:", dst)
    print("\tExists:", os.path.exists(dst))

    if os.path.exists(dst):
        print("Found existing model in location")
        onnx = ONNXArgs.load_from_model(dst)
        print("\tProducer:", onnx.producer)
        print("\tMachine:", onnx.machine_uuid)
        print("\tVersion:", onnx.version)
        print("\tCreation:", onnx.creation_date)
        answer = input("Continue [yes|no]> ")
        if answer != "yes":
            print("Abort.")
            exit(0)

    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    shutil.copyfile(src, dst)

    # create a zip of the model output dir in production, make sure to exclude the
    # analysis subdirectory as it might be huge
    name = os.path.basename(src).split(".")[0]
    src = os.path.dirname(src)
    dst = os.path.join("environment", "production", "configs", onnx.machine_uuid)

    copy_folder(src,
                os.path.join(dst, name),
                exclude_folders=["analysis"],
                exclude_files=["tfevents", ".onnx"],
                modify_flux=True)
    print("Deployment done.")


def redeploy():
    print("Which model should be re-deployed?")
    path, onnx = load_existing_onnx()
    type_ = "model" if "model" in path else "support"
    config = os.path.join("environment", "production", "configs", onnx.machine_uuid, type_, "config.yaml")

    src = os.path.join(os.path.dirname(config), f"{type_}.onnx")
    if not os.path.exists(src):
        os.system("comprisk-score -c " + config)

    # test for same model output
    data_in = np.zeros(shape=(1, onnx.input_length, len(onnx.input_names)))
    data_out = np.zeros(shape=(1, onnx.input_length, len(onnx.output_names)))

    # load pre model
    session = InferenceSession(path)
    onnx_output = list(session.run(None, {"x": data_in, "y": data_out}))

    shutil.copyfile(src, path)


def update_pom():
    if os.path.exists("classes"):
        shutil.rmtree("classes")

    # get the total model version (simply by calculating the model version sum)
    models = [f for f in os.listdir("environment/production/models") if "onnx" in f]
    models = [os.path.join("environment/production/models", f) for f in models]
    model_versions = sum([ONNXArgs.load_from_model(f).version for f in models])

    # get the number of model domains
    model_count = len([ONNXArgs.load_from_model(f).machine_uuid for f in models])

    # get the code version
    code = 0
    for package in os.listdir("packages/"):
        if package.startswith("."):
            continue

        with open(os.path.join("packages", package, "version.txt"), "r") as f:
            major, minor = f.read().split(".")
            code += safe_cast(major, int, 1)
            model_versions += safe_cast(minor, int, 0)

    # modify the pom.xml version tag
    tag = "{http://maven.apache.org/POM/4.0.0}"
    xml.etree.ElementTree.register_namespace("", "http://maven.apache.org/POM/4.0.0")
    et = xml.etree.ElementTree.parse("pom.xml")
    root = et.getroot()

    version_child = root.find(tag + "version")
    version = f"{code}.{model_count}.{model_versions}"
    version_child.text = version

    # Write back to file
    et.write("pom.xml")
    print(f"POM updated (v{version}).")


def delete_model():
    print("Which model should be deleted?")
    path, _ = load_existing_onnx()
    answer = input("Continue [yes|no]> ")
    if answer != "yes":
        print("Abort.")
        return None

    os.remove(path)
    print("Done.")


def show_sensors(uuid_str):
    rename = get_sensors(UUID(uuid_str))
    if not rename:
        print("Error. Unable to get sensors.")
        return None

    result = []
    for uuid, name in rename.items():
        result.append((uuid, name))

    for uuid, name in sorted(result, key=lambda x: x[1]):
        print(f"- {uuid}\t# {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Useful tools for development and deployment.")
    parser.add_argument("--deploy", "-d", action="store_true", default=False)
    parser.add_argument("--edit", "-e", action="store_true", default=False)
    parser.add_argument("--redeploy", "-r", action="store_true", default=False)
    parser.add_argument("--sensors", "-s", type=str, default="")
    args = parser.parse_args()

    if args.edit:
        delete_model()

    if args.deploy:
        deploy_model()

    if args.redeploy:
        redeploy()

    if args.sensors:
        show_sensors(args.sensors)

    update_pom()
