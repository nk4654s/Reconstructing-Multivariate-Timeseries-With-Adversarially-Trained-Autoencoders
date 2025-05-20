"""This module contains the ConfigManager and dict tools to navigate
a nested dict."""

import operator
import os
from copy import deepcopy
from functools import reduce
from typing import Any

import yaml
from optuna import Trial

from Comprisk.data import DataModule
from Comprisk.entrypoints import EntrypointContext
from Comprisk.helper.structure import class_for_name, deepvars
from Comprisk.model import BaseModel
from Comprisk.tune.provider import SuggestionProvider, TunedProvider
from Dataflux import get_logger, logging_mode


logger = get_logger(__name__, path="./logfile", level=logging_mode)


def get_nested_from_dict(a: dict, key: str) -> Any:
    """
    Retrieves the nested value from the dictionary.

    Args:
        a (dict[str, Any]): The dictionary to retrieve the value from.
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


def eval_with_params(a: dict[str, Any], expression: str) -> Any:
    """
    Evaluates an expression with provided parameters.

    Args:
        a (dict[str, Any]): The dictionary of parameters to use in the expression.
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


def set_nested_in_dict(a: dict[str, Any], key: str, value: Any) -> None:
    """
    Sets the nested value in the dictionary.

    Args:
        a (dict[str, Any]): The dictionary to update with the new value.
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
        context (EntrypointContext): The current context.
    """

    def __init__(self, context: EntrypointContext) -> None:
        self._tune = {}
        self._rules = {}

        self._config_folder = context.config_folder
        self._config = deepvars(context.config)
        self._tuning_file = os.path.join(
            context.base_folder, context.environment.value, self._config.get("tuner_config", "")
        )

        if not os.path.exists(self._tuning_file):
            logger.warning("Unable to tune model, could not find config %s", self._tuning_file)
            self.initialized = False
        else:
            self._model_class = class_for_name(*self._config["model"]["class_path"].rsplit(".", maxsplit=1))
            self._model_args = self._config["model"]["init_args"]
            self._data_args = self._config["data"]

            self._post_init()
            self.initialized = True

    @property
    def train_path(self) -> str:
        """
        Returns the path of the original config file used in training.

        Returns:
            str: The path of the original config file used in training.
        """
        return self._config_folder

    def build_model(self) -> BaseModel:
        """
        Finds the model class given its package path and builds the model instance using pre-defined parameters
        of `init_args`.

        Returns:
            BaseModel: The model built.
        """
        class_name = self._config["model"]["class_path"]
        args = self._config["model"]["init_args"]
        class_type = class_for_name(*class_name.rsplit(".", maxsplit=1))
        return class_type(**args)

    def build_datamodule(self) -> DataModule:
        args = self._config["data"]
        return DataModule(**args)

    def _post_init(self) -> None:
        """
        Loads a YAML configuration file and stores it in self._tune.

        This method loads the contents of the specified configuration file,
        which is expected to be in YAML format. The loaded configuration settings
        are stored in self._tune for future use.
        """
        with open(self._tuning_file, "r", encoding="UTF-8") as conf:
            self._tune = yaml.safe_load(conf)

            self._rules = self._tune.get("rules", {})
            del self._tune["rules"]

            self.check_config(self._model_args, self._tune.get("model", {}))
            self.check_config(self._data_args, self._tune.get("data", {}))

    def check_config(self, a: dict[str, Any], b: dict[str, Any]) -> None:
        """
        Check that config file 'b' is a subset of config file 'a'.

        Args:
            a (dict[str, Any]): The base configuration.
            b (dict[str, Any]): The configuration to check against the base.

        Raises:
            AssertionError: If any key in 'b' is not found in 'a'.
        """
        if b is None:
            return

        for key in b:
            assert key in a, f"Unknown key '{key}' in config file."
            if isinstance(a[key], dict):
                self.check_config(a[key]["init_args"], b[key])

    def _suggest(
        self,
        provider: SuggestionProvider,
        train: dict[str, Any],
        tune: dict[str, Any],
        root_key: str = None,
    ) -> dict:
        tune = tune or {}
        train = train or {}

        suggestion = {}
        # build the hyperparameters
        for key in train:
            if isinstance(train[key], dict) and key in tune:
                suggestion[key] = {"class_path": train[key]["class_path"]}
                suggestion[key]["init_args"] = self._suggest(
                    provider, train[key].get("init_args", {}), tune.get(key, {}), key
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

    def apply_rules(self, suggestion: dict[str, Any]) -> dict:
        """
        Apply rules from self._rules to the suggestion dictionary.

        Args:
            suggestion (dict[str, Any]): The suggestion dictionary to modify.

        Returns:
            dict: The modified suggestion dictionary after applying all rules.
        """
        if not self._rules:
            return suggestion

        for rule in self._rules:
            key = list(rule.keys())[0]
            value = eval_with_params(suggestion, rule[key])
            set_nested_in_dict(suggestion, key, value)
        return suggestion

    def init_submodules(self, params: dict) -> dict:
        """
        Initializes submodules based on the provided parameters.

        This method iterates over the keys in the provided dictionary and
        initializes any non-dictionary values it finds. The initialized value
        is then stored in a new dictionary with the same key.

        Args:
            params (dict): A dictionary containing the configuration settings.

        Returns:
            dict: A dictionary with the initialized submodules.
        """
        for key in params:
            if not isinstance(params[key], dict):
                continue

            class_str = params[key]["class_path"].rsplit(".", maxsplit=1)
            init_args = params[key]["init_args"]
            module = class_for_name(*class_str)(**init_args)
            params[key] = module

        return params

    def modify_config(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        This method modifies the original configuration with the provided parameters
        with respect to the rule set.

        Args:
            params (dict): A dictionary containing the configuration settings. Should
                be the dictionary containing the tuned hyperparameters.

        Returns:
            dict: A dictionary containing the modified configuration settings.
        """
        provider = TunedProvider(params)

        base_config = deepcopy(self._config)
        params = self._suggest(provider, base_config["model"]["init_args"], self._tune["model"])
        params = self.apply_rules(params)
        base_config["model"]["init_args"] = params

        base_config["data"] = self._suggest(provider, base_config["data"], self._tune.get("data", {}))
        return base_config

    def suggest(self, trial: Trial) -> tuple[dict, DataModule]:
        """
        Suggest a set of hyperparameters for a given trial.

        Args:
            trial (Trial): The current trial.

        Returns:
            dict: A dictionary containing the suggested hyperparameters.
        """
        provider = SuggestionProvider(trial)

        params = self._suggest(provider, self._model_args, self._tune.get("model", {}))
        params = self.apply_rules(params)

        data_params = self._suggest(provider, self._data_args, self._tune.get("data", {}))
        return self.init_submodules(params), DataModule(**data_params)
