"""This module contains the AnomalyDetector class."""
import os
import importlib
from typing import Dict
import yaml
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from Dataflux.pytorch import Dataset
from Dataflux.helper import get_logger
from Comprisk.helper import get_loss_by_name
from Comprisk.model import BaseModel

logger = get_logger(__name__)


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
    """

    def __init__(self, model_config_path: str, loss: str = "MSELoss") -> None:
        self._model = self._load_model(model_config_path)
        self._device = self._model.device

        self._threshold = 0.0
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
        model = cls.load_from_checkpoint(os.path.join(folder, checkpoints[-1]))
        return model

    def train_threshold(
        self, dataset: Dataset, limit_samples: int = 1_000, overlap: int = None
    ) -> None:
        """This method tunes the threshold based on a given set of samples. The
        reconstruction loss is calculated first, followed by an outlier detection
        using the `Interquartile range (IQR)`. The smallest outlier defines
        the threshold.

        Args:
            dataset (Dataset): The dataset used to train the thresholds.
            limit_samples (int): The amount of samples to use for training the
                threshold.
            overlap (int): Set to a value to define the sliding, this might result
                in overlapping samples.
        """
        logger.debug("Start training the threshold using IQR")
        losses = []
        sliding = overlap if overlap else dataset.get_meta().num_input_sequence

        # apply model on dataset and record loss
        for i in tqdm(range(0, limit_samples * sliding, sliding)):
            sensors, weather, _ = dataset[i]
            loss = self.detect({"sensors": sensors, "weather": weather})
            losses.append(loss)

        df = pd.DataFrame({"losses": losses})

        # Calculate the interquartile range (IQR)
        q1 = df.quantile(0.0)
        q3 = df.quantile(0.9)
        iqr = q3 - q1

        # Define a threshold for outliers
        threshold = 1.5 * iqr

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

    def detect(self, inputs: Dict[str, np.array]) -> float:
        """This method sends the input array to the model, compresses it
        and reconstructs the compressed representation. Then, the
        reconstruction loss is calculated and returned.

        Args:
            inputs (np.array): The array of which the reconstruction loss
                should be calculated.

        Returns:
            float: The reconstruction loss.
        """
        # create tensor of inputs and move it to the same device as
        # the model
        inputs["sensors"] = torch.tensor(inputs["sensors"], device=self._device)
        inputs["weather"] = torch.tensor(inputs["weather"], device=self._device)
        inputs["sensors"] = torch.unsqueeze(inputs["sensors"], dim=0)
        inputs["weather"] = torch.unsqueeze(inputs["weather"], dim=0)

        # pass the inputs to the model, compress it using the autoencoder
        # then restore the compressed representation
        reconstruct = self._model.predict(inputs)

        # calculate the loss of reconstruction, high means anomaly detected
        reconstruction_loss = self._loss(inputs["sensors"], reconstruct)
        return reconstruction_loss.detach().cpu().numpy()

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
        reconstruction_loss = self.detect(inputs)
        if not return_loss:
            return reconstruction_loss >= self._threshold

        return reconstruction_loss >= self._threshold, reconstruction_loss
