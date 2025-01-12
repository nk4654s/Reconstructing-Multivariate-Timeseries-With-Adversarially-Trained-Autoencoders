import os
import shutil
import subprocess
from typing import Dict, List
from argparse import ArgumentParser

import yaml



def model_folder_structure(path: str = "models", structure: List[str] = []) -> List[str]:
    """
    This function creates a list of folders that contains the model's config file.

    Args:
        path (str): The path to the model's folder. Defaults to 'models'.
        structure (List): The list of folders that contains the model's config file. Defaults to [].
    
    Returns:
        List[str]: The list of folders that contains the model's config file.
    """
    entities = os.listdir(path)

    # check if the folder contains the model's comfig. If so, then the folder is
    # a model's output folder
    if "config.yaml" in entities:
        return structure + [path]

    # go a level deeper
    entity_paths = [os.path.join(path, entity) for entity in entities]
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
    parser.add_argument("--load", "-l", type=str, help="Load a config describing the pipeline.")
    args = parser.parse_args()

    # load the pipeline config
    with open(args.config, "r", encoding="utf-8") as f:
        pipeline = yaml.load(f, Loader=yaml.BaseLoader)

    # get the environment, defaults to "dev"
    env = pipeline.get("environment", "dev")
    
    # download the dataset, or copy the configuration to the desired output folder
    # this can only happen after the model was trained, so hold back this information
    dataset_config: List[Dict] = pipeline.get("dataset", [])
    copy_config = []
    for config in dataset_config:
        if config.get("copy_to_output", None):
            copy_config.append(config)
            continue

        subprocess.run(["flux-pipeline", "-c", os.path.join("environment", env, config["config"])])

    # tune the model, this will not have an effect on the model's output folder
    tuner = pipeline.get("tuner", None)
    tuned_config: str | None = None
    if tuner:
        subprocess.run(["comprisk-tuner", "-c", os.path.join("environment", env, tuner["config"]), "--tuner-config",
                        os.path.join("environment", env, tuner["tuner_config"]), "--tuner-trials", tuner["tuner_trials"]])
        tuned_config = find_tuned_config(os.path.join("environment", env, tuner["config"]))

    # train the model
    trainer_list: List[Dict] = pipeline.get("trainer", [])
    latest_config = args.load if args.load else ""
    recently_added = ""
    for i, trainer in enumerate(trainer_list):
        # store the current folder structure of the model output root folder
        # after training, a new folder will be created, which must be passed to
        # the analyzer node
        initial_structure = model_folder_structure(path=os.path.join("environment", env, "models"))

        # use the tuned config if available and wanted, this only applies for the first
        # trainer. Each next trainer will use the one of the previously trained model
        config: str = trainer.pop("config")
        prefer_tuned = trainer.pop("prefer_tuned", False)
        if tuned_config and prefer_tuned:
            config = tuned_config

        # train the model
        args = ["comprisk-trainer", "-c", os.path.join("environment", env, config)]
        args += ["--finetune", latest_config] if latest_config else []
        for key, value in trainer.items():
            args += [f"--{key}", value]

        subprocess.run(args)

        # find the output folder of the recently trained model
        modified_structure = model_folder_structure(path=os.path.join("environment", env, "models"))
        recently_added = find_recent_folders(initial_structure, modified_structure)[0]
        latest_config = os.path.join(recently_added, "config.yaml")

    # copy the configs (if existing) to the recently added model output folder
    for config in copy_config:
        source = os.path.join("environment", env, config["config"])
        destin = os.path.join(recently_added, config["copy_to_output"])
        shutil.copy(source, destin)

    # train the scoring transformer
    score_train = pipeline.get("score_trainer", None)
    if score_train:
        args = ["comprisk-train-score", "-c", latest_config]
        for key, value in score_train.items():
            args += [f"--{key}", value]
        subprocess.run(args)

    # analyze the model
    analyzer_list = pipeline.get("analyzer", [])
    for analyzer in analyzer_list:
        command = ["comprisk-analyzer", "-c", latest_config, "--dataset",
                    os.path.join("environment", env, analyzer["dataset"]), "--prefix", analyzer["prefix"]]

        subprocess.run(command)    
    
    # generate the new score values and upload them
    generate_config = pipeline.get("generate", None)
    if generate_config and recently_added:
        args = ["comprisk-generate", "-c", latest_config, "--dataset", generate_config["config"]]
        subprocess.run(args)

if __name__ == "__main__":
    cli_main()
