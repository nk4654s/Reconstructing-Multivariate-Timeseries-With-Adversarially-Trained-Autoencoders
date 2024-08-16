"""This module contains the AnomalyDetector class."""

from enum import Enum
import os
import importlib
from typing import Dict
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt

from Dataflux.pytorch import Dataset
from Dataflux.pytorch.sampler import RandomBatchSampler
from Dataflux.pytorch.utils import collate_fn

from Dataflux.helper import get_logger
from Comprisk.helper import get_loss_by_name, init_sub_modules
from Comprisk.model import BaseModel
from Comprisk import logger


class AnomalyDetectionMethod(Enum):
    """The anomaly detection methods implemented in the AnomalyDetector
    class.
    """
    QUANTILE = "quantile"
    MEAN = "mean"


class AnomalyDetector:
    """The detector class tries to find an anomaly in a given input by passing the
    data into an auto-encoder. here, the input is compressed and reconstructed, while
    the reconstruction loss can give a hint about unseen behaviour (anomaly). The
    threshold above which the reconstruction loss counts as anomaly can be set trained
    or set manually.

    Usage:
    ```python
    detector = AnomalyDetector("./path/to/model/conf.yml")
    detector.train_threshold(dataset)
    if detector.is_anomaly(inputs_to_test):
        print("anomaly detected)
    ```

    Args:
        model_config_path (str): The path to the model configuration. The model's
            checkpoints must be in the same folder.
        loss (str): The loss function to use. Defaults to 'MSELoss'.
        detection_method (AnomalyDetectionMethod): The method used to detect anomalies. Defaults
        to AnomalyDetectionMethod.MEAN.
    """

    def __init__(self, model_config_path: str, loss: str = "MSELoss",
                 detection_method: AnomalyDetectionMethod = AnomalyDetectionMethod.MEAN) -> None:
        self._model = self._load_model(model_config_path)
        self._device = self._model.device
        self._detection = detection_method

        self._threshold = 0.0
        self._loss_name = loss
        self._loss = get_loss_by_name(loss)

    def _load_model(self, path: str) -> BaseModel:
        # open the config of the trained model
        with open(path, "r", encoding="UTF-8") as stream:
            config = yaml.safe_load(stream)

        # get the model's class
        class_str = config["model"]["class_path"].rsplit(".", maxsplit=1)
        module = importlib.import_module(class_str[0])
        cls = getattr(module, class_str[1])

        # load the model from the latest checkpoint
        folder = os.path.dirname(path)
        checkpoints = sorted([_ for _ in os.listdir(folder) if "ckpt" in _])
        sub_modules = init_sub_modules(folder)
        model = cls.load_from_checkpoint(
            os.path.join(folder, checkpoints[-1]), **sub_modules
        )
        model.eval()
        return model

    @property
    def model(self) -> BaseModel:
        """Returns the loaded model."""
        return self._model

    @property
    def threshold(self) -> float:
        """Simple getter of the internal threshold value. Make sure to run 
        `self.train_threshold`, otherwise the `self.threshold` value might be incorrect.
        """
        return self._threshold

    def set_threshold(self, threshold: float) -> None:
        """Setter of the threshold.

        Args:
            threshold (float): The threshold over which a loss counts as an anomaly.
        """
        self._threshold = threshold

    def train_threshold(
        self,
        dataset: Dataset,
        limit_samples: int = 1_000,
        batch_size: int = 32,
        visualize: bool = False,
        save: bool = None,
    ) -> None:
        """This method tunes the threshold based on a given set of samples. The
        reconstruction loss is calculated first, followed by an outlier detection
        using the `Interquartile range (IQR)`. The smallest outlier defines
        the threshold.

        Args:
            dataset (Dataset): The dataset used to train the thresholds.
            limit_samples (int): The amount of samples to use for training the
                threshold.
            batch_size (int): The batch size of the input. Defaults to 32.
            visualize (bool): Wether to visualize the outlier search or not.
            save (bool): Wether to visualize and save the outlier search or not. This
                will overwrite visualize, but it will not show the visualization. Provide
                the path where to store the image.
        """
        self._threshold = 1
        logger.debug("Start training the threshold using IQR")
        losses = []

        # apply model on dataset and record loss
        batch_sampler = RandomBatchSampler(len(dataset), batch_size=batch_size)
        loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
        data_iter = loader.__iter__()

        for _ in trange(0, limit_samples // batch_size, 1, leave=True):
            loss = self.detect(next(data_iter))
            losses += list(loss)

        df = pd.DataFrame({"losses": losses})

        # Calculate mean and std
        if self._detection == AnomalyDetectionMethod.MEAN:
            mean = df.mean()
            std = df.std()
            sigma = std * 3

            # Identify the outliers, which are defining the threshold
            outliers = (
                df[(df < (mean - sigma)) | (df > (mean + std))].dropna().to_numpy()
            )
        # Calculate the interquartile range (IQR)
        if self._detection == AnomalyDetectionMethod.QUANTILE:
            q1 = df.quantile(0.0)
            q3 = df.quantile(0.9)
            iqr = q3 - q1

            # # Define a threshold for outliers
            threshold = 2 * iqr

            # Identify the outliers, which are defining the threshold
            outliers = (
                df[(df < (q1 - threshold)) | (df > (q3 + threshold))].dropna().to_numpy()
            )
        if len(outliers) > 0:
            self._threshold = outliers.min()
            logger.debug(
                "Found %s outliers, setting threshold to %s",
                len(outliers),
                self._threshold,
            )
        else:
            self._threshold = max(losses)
            logger.debug(
                "No outliers detected, fallback to max loss %s", self._threshold
            )

        # extract the outlier indicies
        if visualize or save:
            _losses = np.array(losses)
            idx = np.where(_losses > self.threshold)[0]
            idx = list(idx - 1) + list(idx) + list(idx + 1)
            idx = sorted(idx)
            mask = np.ones_like(losses, dtype=np.bool_)
            mask[idx] = False

            # build the plot
            plt.title("Reconstruction Loss (Random Samples)")

            # plot the outliers
            _losses = np.array(losses)
            _losses[mask] = np.nan
            plt.plot(_losses, color="teal", alpha=0.5, label="Anomalies")

            # plot all non-outliers
            _losses = np.array(losses)
            _losses[_losses >= self.threshold] = np.nan
            plt.plot(_losses, color="gray", alpha=0.2, label="Reconstruction Loss")

            # plot the threshold line
            plt.plot(
                [self.threshold for _ in range(len(_losses))],
                color="gray",
                alpha=0.6,
                label="Threshold",
            )
            plt.xlabel("# Samples")
            plt.ylabel(self._loss_name)
            plt.legend(loc="lower right")

            if visualize:
                plt.show()

            if save:
                plt.savefig(save)

    def reconstruction(
        self, inputs: Dict[str, np.array], detach: bool = True
    ) -> np.array:
        """Calculate the reconstruction of the input data.

        Args:
            inputs (Dict): The input dictionary, which contains the original data.
            detach (bool): Wether to detach the tensor or not. Defaults to true.

        Returns:
            np.array: Either the array or tensor, depending on `detach`.
        """
        # create tensor of inputs and move it to the same device as
        # the model
        if len(inputs["sensors"].shape) != 3:
            inputs["sensors"] = torch.tensor(inputs["sensors"], device=self._device)
            inputs["weather"] = torch.tensor(inputs["weather"], device=self._device)
            inputs["sensors"] = torch.unsqueeze(inputs["sensors"], dim=0)
            inputs["weather"] = torch.unsqueeze(inputs["weather"], dim=0)
        else:
            inputs["sensors"] = inputs["sensors"].to(self._device)
            inputs["weather"] = inputs["weather"].to(self._device)

        # pass the inputs to the model, compress it using the autoencoder
        # then restore the compressed representation
        reconstruct = self._model.predict(inputs)
        if not detach:
            return reconstruct, inputs
        return reconstruct.detach().cpu().numpy()

    def detect(self, inputs: Dict[str, np.array], feature: bool = False) -> float:
        """This method sends the input array to the model, compresses it
        and reconstructs the compressed representation. Then, the
        reconstruction loss is calculated and returned.

        Args:
            inputs (np.array): The array of which the reconstruction loss
                should be calculated.
            feature (bool): If set to true, the loss per feature will be returned.
                Otherwise the MSELoss of all features will be returned.

        Returns:
            float: The reconstruction loss.
        """
        assert (
            self._threshold > 0.0
        ), "Define a threshold first, either manually or using self.train_threshold()"

        # pass the inputs to the model, compress it using the autoencoder
        # then restore the compressed representation
        reconstruct, inputs = self.reconstruction(inputs, detach=False)

        # calculate the loss of reconstruction, high means anomaly detected
        if not feature:
            losses = []
            for i in range(reconstruct.shape[0]):
                reconstruction_loss = self._loss(inputs["sensors"][i], reconstruct[i])
                losses.append(reconstruction_loss.detach().cpu().numpy())

            return losses

        per_feature = []
        for i in range(reconstruct.shape[-1]):
            x = torch.unsqueeze(inputs["sensors"][:, i], dim=-1)
            x_hat = torch.unsqueeze(reconstruct[:, i], dim=-1)
            per_feature.append(self._loss(x, x_hat).detach().cpu().numpy())
        return per_feature

    def is_anomaly(
        self, inputs: Dict[str, np.array], return_loss: bool = False
    ) -> bool:
        """This method employs the `self.detect` method of this class to
        calculate the reconstruction loss. Afterward, the loss will be compared
        against the threshold in this class.

        Args:
            inputs (Dict[str, np.array]): The arrays of which the reconstruction loss
                should be calculated.
            return_loss (bool): If true, this method returns the bool of anomaly and
                the loss value itself.

        Returns:
            bool: True if an anomaly was detected, otherwise False.
        """
        reconstruction_loss = np.array(self.detect(inputs))
        if not return_loss:
            return (reconstruction_loss >= self._threshold).any()

        return (reconstruction_loss >= self._threshold).any() >= self._threshold, reconstruction_loss
