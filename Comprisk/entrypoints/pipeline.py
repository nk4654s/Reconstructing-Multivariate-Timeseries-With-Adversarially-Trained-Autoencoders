import os
import subprocess
from typing import List
from argparse import ArgumentParser

import yaml

from Comprisk.entrypoints.trainer import train_score_transformer


def model_folder_structure(path: str = "models", structure: List[str] = []) -> List[str]:
    """
    This function creates a list of folders that contains the model's config file.

    Args:
        path (str): The path to the model's folder. Defaults to 'models'.
        structure (List): The list of folders that contains the model's config file. Defaults to [].
    
    Returns:
        List[str]: The list of folders that contains the model's config file.
    """
    entites = os.listdir(path)

    # check if the folder contains the model's comfig. If so, then the folder is
    # a model's output folder
    if "config.yaml" in entites:
        return structure + [path]

    # go a level deeper
    entity_paths = [os.path.join(path, entity) for entity in entites]
    for entity in entity_paths:
        if os.path.isdir(entity):
            structure = model_folder_structure(entity, structure)
    
    # all folder's searched, return the list of model output folders
    return structure


def find_recent_folders(a: List[str] = [], b: List[str] = []) -> List[str]:
    """
    This function finds the folders that are not in a list of folders and return them as a new.

    Args:
        a (List): The list of folders that contains the model's config file. Defaults to [].
        b (List): The list of folders that contains the model's config file. Defaults to [].

    Returns:
        List[str]: The list of folders that are new in list b compared to list a.
    """
    return [path for path in b if path not in a]


def find_tuned_config(path: str) -> str:
    """
    This function finds the tuned config in the config's folder.

    Args:
        path (str): The path to the config file, which was not tuned.
    """
    config_name, extension = os.path.basename(path).split(".")
    config_folder = os.path.dirname(path)

    configs = [c for c in os.listdir(config_folder) if config_name in c]
    configs.remove(config_name + "." + extension)

    return os.path.join(config_folder, max(configs))


def cli_main():
    parser = ArgumentParser(description="The Comprisk training pipeline")
    parser.add_argument("--config", "-c", type=str, help="The config describing the pipeline.")
    args = parser.parse_args()

    # load the pipeline config
    with open(args.config, "r", encoding="utf-8") as f:
        pipeline = yaml.load(f, Loader=yaml.BaseLoader)
    
    # download the dataset
    dataset_config = pipeline.get("dataset", [])
    for config in dataset_config:
        subprocess.run(["flux-pipeline", "-c", config["config"]])

    # tune the model, this will not have an effect on the model's output folder
    tuner = pipeline.get("tuner", None)
    tuned_config = None
    if tuner:
        subprocess.run(["comprisk-tuner", "-c", tuner["config"], "--tuner-config",
                        tuner["tuner_config"], "--tuner-trials", tuner["tuner_trials"]])
        tuned_config = find_tuned_config(tuner["config"])

    # train the model
    trainer_list = pipeline.get("trainer", [])
    latest_config = ""
    for i, trainer in enumerate(trainer_list):
        # store the current folder structure of the model output root folder
        # after training, a new folder will be created, which must be passed to
        # the analyzer node
        initial_structure = model_folder_structure()
        
        # use the tuned config if available and wanted, this only applies for the first
        # trainer. Each next trainer will use the one of the previously trained model
        config = trainer["config"]
        if tuned_config and trainer["prefer_tuned"]:
            config = tuned_config
        
        # train the model
        args = ["comprisk-trainer", "-c", config, "--tag", trainer["tag"]]
        args += [] if i == 0 else ["-f", latest_config]
        subprocess.run(args)

        # find the output folder of the recently trained model
        modified_structure = model_folder_structure()
        recently_added = find_recent_folders(initial_structure, modified_structure)[0]
        latest_config = os.path.join(recently_added, "config.yaml")

    # analyze the model
    analyzer_list = pipeline.get("analyzer", [])
    for analyzer in analyzer_list:
        subprocess.run(["comprisk-analyzer", "-c", latest_config, "--dataset",
                        analyzer["dataset"], "--prefix", analyzer["prefix"],
                        "--static-scores", analyzer.get("static_scores", False)])
    
    # generate the new score values and upload them
    generate_config = pipeline.get("generate", None)
    if generate_config:
        subprocess.run(["comprisk-generate", "-m", generate_config["model"], "-d",
                        generate_config["dataset"]])

if __name__ == "__main__":
    cli_main()
