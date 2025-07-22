import argparse
import os.path
import shutil
import xml
from uuid import UUID

import numpy as np
from onnxruntime import InferenceSession

from CompriskV2.model.utils import OnnxArgs
from Comprisk.helper.structure import search_models, copy_folder
from Dataflux.helper import get_sensors


def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        print(f"Invalid input '{val}'.")
        exit(1)


def to_string(onnx) -> str:
    return f"[{onnx.creation_date}] [v{onnx.version}] {onnx.machine_uuid}"


def get_model_type(path: str) -> str:
    name = os.path.basename(path)
    return name.rsplit("_")[-1].replace(".onnx", "")


def model_comp(model):
    args = OnnxArgs.load_from_model(model)
    return args.machine_uuid, args.creation_date, get_model_type(model)


def filter_models(models, filter):
    if not filter:
        return models

    return [m for m in models if (filter.machine_uuid in m)]


def select_model(path: str, calls: int = 0, filter=None):
    if calls == 3 or calls > 0 and input("Exit? [y|n]> ") == "y":
        print("Abort.")
        exit(1)

    print("Select model")
    models = search_models(path)
    models = filter_models(models, filter)
    models = sorted(models, key=model_comp)
    for i, path in enumerate(models):
        model_type = get_model_type(path)
        onnx = OnnxArgs.load_from_model(path)
        print(f"{to_string(onnx)} -> {i} ({model_type})")

    i = safe_cast(input(f"0-{len(models) - 1}> "), int)
    path = models[i]

    onnx = OnnxArgs.load_from_model(path)
    print("Loaded selected model")
    print("\tProducer:", onnx.producer)
    print("\tMachine:", onnx.machine_uuid)
    print("\tVersion:", onnx.version)
    print("\tCreation:", onnx.creation_date)

    if input("Continue? [y|n]> ") != "y":
        return select_model(path, calls + 1)
    return path, onnx


def check_model_deployed(onnx, model_type):
    for path in search_models("environment/production/models"):
        model = OnnxArgs.load_from_model(path)
        if onnx.machine_uuid == model.machine_uuid and get_model_type(path) == model_type:
            return path, model

    return None, None


def deploy_model():
    n_path, n_model = select_model("environment/dev/models")
    n_type = get_model_type(n_path)
    c_path, c_model = check_model_deployed(n_model, n_type)
    if c_path:
        print("Model exists, use `re-deploy` route")
        print(to_string(c_model))
        exit(1)

    print("Deploying selected model")
    print(to_string(n_model))

    dst = os.path.join("environment/production/models", f"{n_model.machine_uuid}_{n_type}.onnx")
    print("\tCopy to:", dst)

    if not os.path.exists("environment/production/models"):
        os.makedirs(os.path.dirname(dst))
    shutil.copyfile(n_path, dst)

    dst = os.path.join("environment/production/configs", n_model.machine_uuid)
    copy_folder(
        os.path.dirname(n_path),
        os.path.join(dst, n_type),
        exclude_folders=["analysis"],
        exclude_files=["tfevents", ".onnx"],
        modify_flux=True,
    )
    print("Deployment done.")


def redeploy():
    print("Which model should be re-deployed?")
    c_path, onnx = select_model("environment/production/models")
    n_path, n_model = select_model("environment/dev/models", filter=onnx)
    n_type = get_model_type(n_path)
    if get_model_type(c_path) != n_type:
        print("Invalid model types")
        exit(0)

    # test for same model output
    data_in = np.zeros(shape=(1, onnx.input_length, len(onnx.input_names)), dtype=np.float32)
    data_out = np.zeros(shape=(1, onnx.output_length, len(onnx.output_names)), dtype=np.float32)

    # compare pre-model and new-model (should be identical in terms of output)
    output_pre = list(InferenceSession(c_path).run(None, {"x": data_in, "y": data_out}))
    output_now = list(InferenceSession(n_path).run(None, {"x": data_in, "y": data_out}))
    if not ((output_pre[0] == output_now[0]).all() and (output_pre[1] == output_now[1]).all()):
        print("The deployed model outputs other values than the newly generated model.")
        answer = input("Continue re-deployment [yes|no]> ")
        if answer != "yes":
            print("Abort.")
            exit(0)

    deployed = os.path.join("environment/production/configs", n_model.machine_uuid, n_type)
    shutil.rmtree(deployed)
    shutil.copyfile(n_path, c_path)

    copy_folder(
        os.path.dirname(n_path),
        deployed,
        exclude_folders=["analysis"],
        exclude_files=["tfevents", ".onnx"],
        modify_flux=True,
    )
    print("Re-deployed model")


def update_pom():
    if os.path.exists("classes"):
        shutil.rmtree("classes")

    # get the total model version (simply by calculating the model version sum)
    models = [f for f in os.listdir("environment/production/models") if "onnx" in f]
    models = [os.path.join("environment/production/models", f) for f in models]
    model_versions = sum([OnnxArgs.load_from_model(f).version for f in models])

    # get the number of model domains
    model_count = len([OnnxArgs.load_from_model(f).machine_uuid for f in models])

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
    path, _ = select_model()
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
