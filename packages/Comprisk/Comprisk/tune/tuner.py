"""This module contains the tuner class based on the `optuna` package to find
optimal hyperparameters for training."""
from typing import Any, Dict
from copy import deepcopy
import os
import optuna
from optuna.trial import Trial

from Comprisk.tune.config import ConfigManager


class Tuner:
    """This class finds the optimal hyperparameters for a model using the `optuna`
    package. This package employs an advanced for of bayesian optimisation to explore
    the hyperparameter space.

    Args:
        cli (CompriskCLI): The comprisk command line interface object, which contains
            the initialized model and datamodule. During optimisation, the model
            will be re-initialized with a different set of hyperparameters.
    """

    def __init__(self, config: ConfigManager) -> None:
        self._model = config.cli.model
        self._datamodule = config.cli.datamodule

        self._trainer = config.cli.trainer
        self._trainer.early_stopping_callback.monitor = "train/MSELoss"

        self._trainer.save_config_callback = None
        self._trainer.logger = False

        self._config = config
        self._hparam_deps = None

    def objective(self, trial: Trial) -> float:
        """The objective function is used by the `self.optimize` method and contains
        the training and validation loop of the model. Here, different hyperparameters
        will be tested to find the lowest loss.

        Args:
            trial (Trial): The trial object managed by `optuna`.
        """
        # define the new hparam suggestion and create the model to test
        suggestion, datamodule = self._config.suggest(trial)
        test_model = self._model.__class__(**suggestion)

        datamodule.setup(stage="fit")

        # train the test model and validate it, the resulting loss is used to
        # set a new hparams suggestion
        trainer = deepcopy(self._trainer)
        trainer.fit(
            test_model, datamodule.train_dataloader(), val_dataloaders=None
        )
        trainer.save_checkpoint(".temp.ckpt")
        result = trainer.validate(
            dataloaders=datamodule.val_dataloader(), ckpt_path=".temp.ckpt"
        )
        loss = result[0]["validate/MSELoss"]
        return loss

    def optimize(self, n_trials: int, tag: str = None) -> Dict[str, Any]:
        """This method starts the optimization process. Make sure to have `self.config` called
        beforehand.

        Args:
            n_trials (int): The number of search iterations before termination.
            tag (str): The tag str as prefix of the experiment name.
        """
        if not os.path.exists("logs"):
            os.makedirs("logs")

        name = (
            f"{self._model.__class__.__name__}-{tag}"
            if tag
            else self._model.__class__.__name__
        )
        study = optuna.create_study(
            storage="sqlite:///logs/optuna.sqlite3",
            study_name=name,
            sampler=optuna.samplers.GPSampler(),
            load_if_exists=True,
            direction="minimize",
        )
        study.optimize(
            self.objective,
            n_trials=n_trials,
            #catch=(RuntimeWarning, TypeError, RuntimeError),
        )

        # restore the datamodel to the training mode
        self._datamodule.setup(stage="fit")
        os.remove(".temp.ckpt")
        return study.best_trial.params
