import os
import shutil
import yaml
import numpy as np

from onnxruntime import InferenceSession
from Dataflux.helper import modify_flux_config
from CompriskV2.model.utils import OnnxArgs


def search_models(path: str, search_depth: int = 50) -> list[str]:
    """Searches ONNX models in a given folder recursively.

    Args:
        path (str): Where to search, subfolders will be scanned as well if `search_depth` > 1:
        search_depth (int): How deep to search recursively. Defaults to 50.

    Returns:
        List[str]: The list of found paths to ONNX models
    """
    if os.path.basename(path) == "derived_from":
        return []

    result = []
    for sub_path in os.listdir(path):
        sub_path = os.path.join(path, sub_path)

        if os.path.isdir(sub_path):
            result += search_models(sub_path, search_depth=search_depth - 1)
            continue

        if sub_path.split(".")[-1] == "onnx":
            result.append(sub_path)

    return result


def copy_folder(
    source: str,
    destination: str,
    exclude_folders: list[str] = None,
    exclude_files: list[str] = None,
    modify_flux: bool = False,
) -> None:
    """
    Copies a folder from the source to the destination, excluding specified subfolders.

    Args:
        source (str): The path to the folder that needs to be copied.
        destination (str): The path where the folder should be copied.
        exclude_folders (list of str): A list of subfolder names to exclude from the copy process.
        exclude_files (list of str): A list of file names (can be substrings) to exclude from the copy process.
        modify_flux (bool): Whether to modify the flux config paths. Defaults to False.

    Returns:
        None
    """
    exclude_folders = exclude_folders or []
    exclude_files = exclude_files or []

    flux_files = ["train.yaml", "validate.yaml"]
    if modify_flux:
        exclude_files += flux_files

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination):
        os.makedirs(destination)

    for root, dirs, files in os.walk(source):
        # Filter out the directories to exclude
        dirs[:] = [d for d in dirs if d not in exclude_folders]

        # Construct the destination path for the current root
        relative_path = os.path.relpath(root, source)
        destination_path = os.path.join(destination, relative_path)

        # Create the destination directory if it doesn't exist
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        # Copy the files to the destination
        for f in files:
            skip = sum([1 for _ in exclude_files if _ in f])
            if skip:
                continue
            source_file_path = os.path.join(root, f)
            destination_file_path = os.path.join(destination_path, f)
            shutil.copy2(source_file_path, destination_file_path)

    if modify_flux:
        for flux in flux_files:
            modify_flux_config(os.path.join(source, flux), os.path.join(destination, flux))

    config = os.path.join(destination, "config.yaml")
    with open(config, "r+") as f:
        config = yaml.safe_load(f)
        config["data"]["train"] = os.path.join(destination, "train.yaml")
        config["data"]["validate"] = os.path.join(destination, "validate.yaml")

        config["env"] = "production"
        yaml.safe_dump(config, f)


def safe_cast(val: str, to_type: type):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        print(f"Invalid input '{val}'.")
        exit(1)


def get_model_type(path: str) -> str:
    name = os.path.basename(path)
    split = name.rsplit("_", maxsplit=1)

    if ".onnx" not in name:
        return None

    return split[-1].replace(".onnx", "")


def model_comp(model):
    args = OnnxArgs.load_from_model(model)
    return args.machine_uuid, args.creation_date, get_model_type(model)


def filter_models(models: list[str], filter: OnnxArgs) -> list[str]:
    if not filter:
        return models

    return [m for m in models if (filter.machine_uuid in m)]


def check_model_deployed(onnx: OnnxArgs, model_type: str, root_dir: str = "environment/production/models"):
    for path in search_models(root_dir):
        model = OnnxArgs.load_from_model(path)
        if onnx.machine_uuid == model.machine_uuid and get_model_type(path) == model_type:
            return path, model

    return None, None


def check_model_consistency(path_a: str, path_b: str):
    onnx = OnnxArgs.load_from_model(path_a)
    data_in = np.zeros(shape=(1, onnx.input_length, len(onnx.input_names)), dtype=np.float32)
    data_out = np.zeros(shape=(1, onnx.output_length, len(onnx.output_names)), dtype=np.float32)

    output_pre = list(InferenceSession(path_a).run(None, {"x": data_in, "y": data_out}))
    output_now = list(InferenceSession(path_b).run(None, {"x": data_in, "y": data_out}))
    return (output_pre[0] == output_now[0]).all() and (output_pre[1] == output_now[1]).all()
