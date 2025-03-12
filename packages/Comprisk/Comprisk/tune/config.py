"""This module contains the ConfigManager and dict tools to navigate
a nested dict."""

import operator
from copy import deepcopy
from functools import reduce
from typing import Any, Dict, Tuple

import yaml
from optuna import Trial

from Comprisk.cli.cli import CompriskCLI
from Comprisk.data import DataModule
from Comprisk.helper.structure import class_for_name, get_config_path
from Comprisk.tune.provider import SuggestionProvider, TunedProvider


def get_nested_from_dict(a: Dict, key: str) -> Any:
    """
    Retrieves the nested value from the dictionary.

    Args:
        a (Dict[str, Any]): The dictionary to retrieve the value from.
        key (str): The key or nested keys used to find the value.

    Returns:
        Any: The retrieved value. If the key is not found, returns None.
    """
    try:
        if len(key.split(".")):
            key = ".init_args.".join(key.split("."))
        return reduce(operator.getitem, key.split("."), a)
    except KeyError:
        return None


def eval_with_params(a: Dict[str, Any], expression: str) -> Any:
    """
    Evaluates an expression with provided parameters.

    Args:
        a (Dict[str, Any]): The dictionary of parameters to use in the expression.
        expression (str): The expression to evaluate using the parameters.

    Returns:
        Any: The result of the evaluated expression.
    """
    for param in expression.split(" "):
        value = get_nested_from_dict(a, param)
        if value is None:
            continue
        expression = expression.replace(param, str(value))
    return eval(expression)


def set_nested_in_dict(a: Dict[str, Any], key: str, value: Any) -> None:
    """
    Sets the nested value in the dictionary.

    Args:
        a (Dict[str, Any]): The dictionary to update with the new value.
        key (str): The key or nested keys used to find and set the value.
        value (Any): The value to be set in the dictionary.

    Returns:
        None
    """
    if len(key.split(".")):
        key = ".init_args.".join(key.split("."))
    keys = key.split(".")
    for key in keys[:-1]:
        a = a.setdefault(key, {})
    a[keys[-1]] = value


class ConfigManager:
    """
    Manages configuration settings.

    This class provides methods to handle configuration-related tasks,
    such as setting and getting values from the configuration dictionary.

    Args:
        cli (CompriskCLI): The CompriskCLI instance associated with this manager.
    """

    def __init__(self, cli: CompriskCLI) -> None:
        self._tune = None
        self._rules = None
        self._cli = cli

        # load the original train config
        self._train_path = get_config_path(cli)
        with open(self._train_path, "r", encoding="utf-8") as f:
            self._base_config = yaml.load(f, Loader=yaml.FullLoader)

        self._model = self._base_config["model"]["init_args"]
        self._data = self._base_config["data"]

    @property
    def train_path(self) -> str:
        """
        Returns the path of the original config file used in training.

        Returns:
            str: The path of the original config file used in training.
        """
        return self._train_path

    @property
    def train_config(self) -> Dict:
        """
        Returns a dictionary containing the initialization arguments.

        This property provides read-only access to the `_train` attribute,
        which contains the configuration settings for the training process.

        Returns:
            Dict: A dictionary with the initialization arguments.
        """
        return self._model

    @property
    def cli(self) -> CompriskCLI:
        """
        Returns an instance of CompriskCLI associated with this ConfigManager.

        This method provides read-only access to the `_cli` attribute,
        which contains the configuration settings for the CLI.

        Returns:
            CompriskCLI: An instance of CompriskCLI.
        """
        return self._cli

    def load(self, path_to_config: str) -> None:
        """
        Loads a YAML configuration file and stores it in self._tune.

        This method loads the contents of the specified configuration file,
        which is expected to be in YAML format. The loaded configuration settings
        are stored in self._tune for future use.

        Args:
            path_to_config (str): The path to the configuration file.
        """
        with open(path_to_config, "r", encoding="UTF-8") as conf:
            self._tune = yaml.safe_load(conf)

        self._rules = self._tune.get("rules", {})
        del self._tune["rules"]

        self.check_config(self._model, self._tune["model"])
        self.check_config(self._data, self._tune.get("data", {}))

    def check_config(self, a: Dict[str, Any], b: Dict[str, Any]) -> None:
        """
        Check that config file 'b' is a subset of config file 'a'.
        
        Args:
            a (Dict[str, Any]): The base configuration.
            b (Dict[str, Any]): The configuration to check against the base.
            
        Raises:
            AssertionError: If any key in 'b' is not found in 'a'.
        """
        for key in b:
            assert key in a, f"Unknown key '{key}' in config file."
            if isinstance(a[key], dict):
                self.check_config(a[key]["init_args"], b[key])

    def _suggest(
        self,
        provider: SuggestionProvider,
        train: Dict[str, Any],
        tune: Dict[str, Any],
        root_key: str = None,
    ) -> None:
        suggestion = {}
        # build the hyperparameters
        for key in train:
            if isinstance(train[key], dict) and key in tune:
                suggestion[key] = {"class_path": train[key]["class_path"]}
                suggestion[key]["init_args"] = self._suggest(
                    provider, train[key]["init_args"], tune[key], key
                )
                continue

            # use hyperparameters from the model config, if they will not be tuned
            if key not in tune:
                suggestion[key] = train[key]
                continue

            # get hyperparameter (spaces) from the tuner config
            value = tune[key]
            name = key if not root_key else f"{root_key}.{key}"
            suggestion[key] = provider.get(name, value)

        return suggestion

    def apply_rules(self, suggestion: Dict[str, Any]) -> Dict:
        """
        Apply rules from self._rules to the suggestion dictionary.
        
        Args:
            suggestion (Dict[str, Any]): The suggestion dictionary to modify.
            
        Returns:
            Dict: The modified suggestion dictionary after applying all rules.
        """
        for rule in self._rules:
            key = list(rule.keys())[0]
            value = eval_with_params(suggestion, rule[key])
            set_nested_in_dict(suggestion, key, value)
        return suggestion

    def init_submodules(self, params: Dict) -> Dict:
        """
        Initializes submodules based on the provided parameters.
        
        This method iterates over the keys in the provided dictionary and 
        initializes any non-dictionary values it finds. The initialized value 
        is then stored in a new dictionary with the same key.
        
        Args:
            params (Dict): A dictionary containing the configuration settings.
        
        Returns:
            Dict: A dictionary with the initialized submodules.
        """
        for key in params:
            if not isinstance(params[key], dict):
                continue

            class_str = params[key]["class_path"].rsplit(".", maxsplit=1)
            init_args = params[key]["init_args"]
            module = class_for_name(*class_str)(**init_args)
            params[key] = module

        return params

    def modify_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method modifies the original configuration with the provided parameters
        with respect to the rule set.

        Args:
            params (Dict): A dictionary containing the configuration settings. Should
                be the dictionary containing the tuned hyperparameters.
        
        Returns:
            Dict: A dictionary containing the modified configuration settings.
        """
        provider = TunedProvider(params)

        base_config = deepcopy(self._base_config)
        params = self._suggest(provider, base_config["model"]["init_args"], self._tune["model"])
        params = self.apply_rules(params)
        base_config["model"]["init_args"] = params

        base_config["data"] = self._suggest(provider, base_config["data"], self._tune.get("data", {}))
        return base_config

    def suggest(self, trial: Trial) -> Tuple[Dict, DataModule]:
        """
        Suggest a set of hyperparameters for a given trial.
        
        Args:
            trial (Trial): The current trial.
            
        Returns:
            Dict: A dictionary containing the suggested hyperparameters.
        """
        provider = SuggestionProvider(trial)

        params = self._suggest(provider, self._model, self._tune["model"])
        params = self.apply_rules(params)

        data_params = self._suggest(provider, self._data, self._tune.get("data", {}))
        return self.init_submodules(params), DataModule(**data_params)
