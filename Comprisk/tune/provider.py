"""This module contains the provider, providing parameters suggestions to the Optuna tuner."""
from typing import Any, Dict

from optuna import Trial


class BaseProvider:
    """
    The BaseProvider defines the structure of a provider used by the ConfigManager of the `tune` sub module.
    The provider is used to provide the hyperparameter values for each trial.
    """
    def get(self, name: str, value_specs: Dict[str, Any]) -> Any:
        """
        The get method suggests a value given the name of the hyperparameter and its value specs.

        Args:
            name (str): The name of the hyperparameter whose value is being suggested.
            value_specs (Dict[str, Any]): A dictionary containing the specifications for the hyperparameter's
                value.
        
        Returns:
            Any: A suggested value for the hyperparameter.
        """
        raise NotImplementedError


class SuggestionProvider(BaseProvider):
    """
    The SuggestionProvider is used to provide the hyperparameter values for each
    trial using Optuna's suggestion methods.

    Args:
        trial (Trial): The Optuna trial object to use for suggestions.
    """
    def __init__(self, trial: Trial) -> None:
        self._trial = trial

    def get(self, name: str, value_specs: Dict[str, Any]) -> Any:
        if value_specs["type"] == "int":
            return self._trial.suggest_int(name, value_specs["min"], value_specs["max"])

        elif value_specs["type"] == "float":
            return self._trial.suggest_float(name, value_specs["min"], value_specs["max"])


class TunedProvider(BaseProvider):
    def __init__(self, params: Dict[str, Any]) -> None:
        self._tuned_values = params

    def get(self, name: str, value_specs: Dict[str, Any]) -> Any:
        return self._tuned_values[name]
