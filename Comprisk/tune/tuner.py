"""This module contains the tuner class based on the `optuna` package to find
optimal hyper parameters for training."""
from typing import Any, Dict
import yaml
import optuna
from optuna.trial import Trial

from Comprisk.cli.cli import CompriskCLI


class Tuner:
    """This class finds the optimal hyper parameters for a model using the `optuna`
    package. This package employs an advanced for of bayesian optimistaion to explore
    the hyper parameter space. The loaded datamodule heavily limits the samples to train
    for faster training.

    Args:
        cli (CompriskCLI): The comrpisk command line interface object, which contains
            the initialized model and datamodule. During optimization, the model
            will be re-initialized with a different set of hyper parameters.
    """
    def __init__(self, cli: CompriskCLI) -> None:
        self._model = cli.model
        self._original_params = cli.model.hparams
        self._datamodule = cli.datamodule
        self._trainer = cli.trainer
        self._trainer.logger = False

        self._suggestions = None

        # heavily reduce the training/validation samples as the tuner
        # just tries to find a parameter suggestion. Longer trainings
        # might result in superior hyperparameters, but the search
        # will take dramatically longer
        self._datamodule.setup(stage="fit", train_size=0.01, val_size=0.002)

    def objective(self, trial: Trial) -> float:
        """The objective function is used by the `self.optimize` method and contains
        the training and validation loop of the model. Here, different hyper
        parameters will be tested to find the lowest loss.

        Args:
            trial (Trial): The trial object managed by `optuna`.
        """
        # define the new hparam suggestion and create the model to test
        suggestion = self._define_suggestions(trial, self._original_params)
        test_model = self._model.__class__(**suggestion)

        # train the test model and validate it, the resulting loss is used to
        # set a new hparams suggestion
        self._trainer.fit(
            test_model, self._datamodule.train_dataloader(), val_dataloaders=None
        )
        result = self._trainer.validate(test_model, self._datamodule.val_dataloader())
        loss = result[0]["validate/MSELoss"]

        # trial.report(loss, 0)
        return loss

    def _define_suggestions(self, trial: Trial, params: Dict) -> Dict[str, Any]:
        suggestion = {}
        # build the hyper parameters
        for key in params.keys():
            # disable dropouts, as the training will be short
            if "dropout" in key:
                suggestion[key] = 0.0
                continue

            # use hyper parameters from the model config, if they will not be tuned
            if key not in self._suggestions.keys():
                suggestion[key] = params[key]
                continue

            # get hyper parameter (spaces) from the tuner config
            value = self._suggestions[key]
            if value["type"] == "int":
                suggestion[key] = trial.suggest_int(key, value["min"], value["max"])

            elif value["type"] == "float":
                suggestion[key] = trial.suggest_float(key, value["min"], value["max"])
        return suggestion

    def config(self, path_to_config: str) -> None:
        """This method loads the tune config from a given path. The `CompriskCLI` provides
        the argument `--tuner-config` which should be the path of the tuner config. This config
        must be a yaml file containg the hyper parameters, their type, and min/max values.

        Args:
            path_to_config (str): The tuner config path.
        """
        with open(path_to_config, "r", encoding="UTF-8") as conf:
            self._suggestions = yaml.safe_load(conf)

        for key in self._suggestions.keys():
            assert key in self._model.hparams.keys(), f"Unexpected parameter '{key}'"

    def optimize(self, n_trials: int) -> Dict[str, Any]:
        """This method starts the optimization process. Make sure to have `self.config` called
        beforehand.

        Args:
            n_trials (int): The number of search iterations before termination.
        """
        assert self._suggestions, "Define a config with hparams to tune first."
        study = optuna.create_study(
            storage="sqlite:///optuna.sqlite3",
            study_name=self._model.__class__.__name__,
            load_if_exists=True,
            direction="minimize",
        )
        study.optimize(
            self.objective,
            n_trials=n_trials,
            catch=(RuntimeWarning, TypeError, RuntimeError),
        )

        # restore the datamodel to the training mode
        self._datamodule.setup(stage="fit")
        return study.best_trial.params
