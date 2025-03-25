import inspect
import os
from typing import Any, Dict, List

import yaml

from Dataflux.helper.utils import class_for_name


def build_config(path: str, config: Dict = None) -> Dict:
    """This function finds the classes defined in the config.
    
    Args:
        path (str): Path to the config file.
        config (Dict): Config to use instead of the file, defaults to empty dict.
    
    Returns:
        Dict[str, Any]: The config as dictionary with the class types instead of
            the strings for the class paths defined with the key 'class_path'.
    """
    if path is None or not os.path.isfile(path):
        return {}

    if not config:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    for key in config:
        if key not in ["modules"]:
            continue

        for i, module_dict in enumerate(config.get(key, [])):
            module = class_for_name(*module_dict["class_path"].rsplit(".", maxsplit=1))
            config[key][i]["class_path"] = module

    return config


def get_module_config_args(config: Dict, module) -> Dict:
    """This function finds the module and its initialization arguments from the
    config.

    Args:
        config (Dict): The config as dictionary with the class types and a subdict
            'init_args' with the initialization arguments of each module.
        module: The module which should be initialized.
    
    Returns:
        Dict[str, Any]: The initialization arguments of the module as dictionary.
    """
    modules = [c for c in config.get("modules", []) if module == c["class_path"]]
    if len(modules) == 0:
        return {}
    
    return modules[0].get("init_args", {})


def expand_module_list(modules: List, config: Dict[str, Any]) -> List:
    """This function expands the list of modules to include all possible modules.

    Args:
        modules (List): The default processing modules defined in the pipeline.
        config (Dict): The config as dictionary with the class types and a subdict
            'init_args' with the initialization arguments of each module.
        
    Returns:
        List: List of worker class configurations
    """
    # add modules from the config file
    extended_modules = []
    for module in config.get("modules", []):
        if "init_args" not in module:
            module["init_args"] = {}
        extended_modules.append(module)

    # add default arguments from the program
    for module in reversed(modules):
        if module in [c["class_path"] for c in config.get("modules", [])]:
            continue

        init_args = {}
        signature = inspect.signature(module.__init__).parameters
        for attr in signature:
            if attr == "self":
                continue
            
            init_args[signature[attr].name] = init_args.get(signature[attr].name, signature[attr].default)
        extended_modules.insert(0, {"class_path": module, "init_args": init_args})
    
    return extended_modules
