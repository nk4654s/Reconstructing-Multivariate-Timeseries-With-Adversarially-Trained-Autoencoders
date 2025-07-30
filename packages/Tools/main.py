import os
import shutil
import xml
import argparse
import onnxruntime
from uuid import UUID

from Dataflux.helper import get_sensors
from CompriskV2.model.utils import OnnxArgs
from Tools import utils


def select_model(path: str, calls: int = 0, filter=None):
    if calls == 3 or calls > 0 and input("Exit? [y|n]> ") == "y":
        print("Abort.")
        exit(1)

    print("Select model")
    models = utils.search_models(path)
    models = utils.filter_models(models, filter)
    models = sorted(models, key=utils.model_comp)
    for i, path in enumerate(models):
        model_type = utils.get_model_type(path)
        onnx = OnnxArgs.load_from_model(path)
        print(f"{onnx.to_string()} -> {i} ({model_type})")

    i = utils.safe_cast(input(f"0-{len(models) - 1}> "), int)
    path = models[i]

    onnx = OnnxArgs.load_from_model(path)
    print("Loaded selected model")
    print("\tProducer:", onnx.producer)
    print("\tMachine:", onnx.machine_uuid)
    print("\tVersion:", onnx.version)
    print("\tCreation:", onnx.creation_date)

    if input("Continue? [y|n]> ") != "y":
        return utils.select_model(path, calls + 1)
    return path, onnx


def deploy_model():
    n_path, n_model = select_model("environment/dev/models")
    n_type = utils.get_model_type(n_path)
    c_path, c_model = utils.check_model_deployed(n_model, n_type)
    if c_path:
        print("Model exists, use `re-deploy` route")
        print(c_model.to_string())
        exit(1)

    print("Deploying selected model")
    print(n_model.to_string())

    dst = os.path.join(
        "environment/production/models", f"{n_model.machine_uuid}_{n_type}.onnx"
    )
    print("\tCopy to:", dst)

    if not os.path.exists("environment/production/models"):
        os.makedirs(os.path.dirname(dst))
    shutil.copyfile(n_path, dst)

    dst = os.path.join("environment/production/configs", n_model.machine_uuid)
    utils.copy_folder(
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
    print(n_path, utils.get_model_type(n_path))
    n_type = utils.get_model_type(n_path)
    if utils.get_model_type(c_path) != n_type:
        print("Invalid model types")
        exit(0)

    # test for same model output
    if not utils.check_model_consistency(c_path, n_path):
        print("The deployed model outputs other values than the newly generated model.")
        answer = input("Continue re-deployment [yes|no]> ")
        if answer != "yes":
            print("Abort.")
            exit(0)

    deployed = os.path.join(
        "environment/production/configs", n_model.machine_uuid, n_type
    )
    shutil.rmtree(deployed)
    shutil.copyfile(n_path, c_path)

    utils.copy_folder(
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
            code += utils.safe_cast(major, int)
            model_versions += utils.safe_cast(minor, int)

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


def info(path: str):
    models = [path] if path else utils.search_models("environment/production/models")
    for model in models:
        onnx = OnnxArgs.load_from_model(model)
        session = onnxruntime.InferenceSession(model)
        output_labels = [o.name for o in session.get_outputs()]

        print("Loaded model for", onnx.machine_uuid)
        print("\tCreation:", onnx.creation_date)
        print("\tVersion:", onnx.version)
        print("\tProducer:", onnx.producer)
        print("\tType:", model.split("_")[-1].replace(".onnx", ""))
        print("\tValue Corridor:", onnx.value_corridor)
        print("\tOutput Labels: ", output_labels)
        print()

    exit(0)


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


def main():
    parser = argparse.ArgumentParser(
        description="Useful tools for development and deployment."
    )
    parser.add_argument("--deploy", "-d", action="store_true", default=False)
    parser.add_argument("--redeploy", "-r", action="store_true", default=False)
    parser.add_argument("--info", "-i", type=str, default=None)
    parser.add_argument("--sensors", "-s", type=str, default="")
    args = parser.parse_args()

    if args.info is not None:
        info(args.info)

    if args.deploy:
        deploy_model()

    if args.redeploy:
        redeploy()

    if args.sensors:
        show_sensors(args.sensors)

    update_pom()


if __name__ == "__main__":
    main()
