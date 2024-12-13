import os
from typing import Any, Dict, List

import yaml

from Dataflux.helper.utils import class_for_name
from Dataflux.modules.module import BasePreProcessor


def build_config(path: str) -> Dict:
    """This function finds the classes defined in the config given the keys
    'preprocessing' and 'postprocessing'. 
    
    Args:
        path (str): Path to the config file.
    
    Returns:
        Dict[str, Any]: The config as dictionary with the class types instead of
            the strings for the class paths defined with the key 'class_path'.
    """
    if path is None:
        return {}

    assert os.path.isfile(path), f"{path} is not a file."
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key, modules in config.items():
        if key not in ["preprocessing", "postprocessing"]:
            continue

        for i, module_dict in enumerate(config.get(key, [])):
            module = class_for_name(*module_dict["class_path"].rsplit(".", maxsplit=1))
            config[key][i]["class_path"] = module

    return config


def get_module_config_args(config: Dict, key, module) -> Dict:
    """This function finds the module and its initialization arguments from the
    config.

    Args:
        config (Dict): The config as dictionary with the class types and a subdict
            'init_args' with the initialization arguments of each module.
        key (str): The key of the modules in the config. Either 'preprocessing' or
            'postprocessing'.
        module (BasePreProcessor): The module which should be initialized.
    
    Returns:
        Dict[str, Any]: The initialization arguments of the module as dictionary.
    """
    modules = [c for c in config.get(key, []) if module == c["class_path"]]
    if len(modules) == 0:
        return {}
    
    return modules[0].get("init_args", {})


def expand_module_list(modules: Dict[str, List[BasePreProcessor]], config: Dict[str, Any]) -> Dict[str, List[BasePreProcessor]]:
    """This function expands the list of modules to include all possible modules.

    Args:
        modules (Dict[str, List[BasePreProcessor]]): The dictionary with the keys 'preprocessing' and
            'postprocessing'.
        config (Dict): The config as dictionary with the class types and a subdict
            'init_args' with the initialization arguments of each module.
        
    Returns:
        Dict[str, List[BasePreProcessor]]: A combination of both input dictionaries with no
            modules duplicated.
    """
    for key in modules.keys():
        for module in config.get(key, []):
            if module["class_path"] not in modules:
                modules[key].append(module["class_path"])
    
    return modules
