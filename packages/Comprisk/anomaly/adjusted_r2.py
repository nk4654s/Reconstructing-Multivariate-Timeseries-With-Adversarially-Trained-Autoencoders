
from tqdm import tqdm
import torch

from lightning import LightningModule
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

import numpy as np

from Dataflux import get_logger, logging_mode

from Comprisk.custom import DataModule
from Comprisk.model import BaseModel
from Dataflux.pytorch import Dataset

logger = get_logger(__name__, "logfile", logging_mode)


class R2ScoreWrapper(LightningModule):
    """
    This score transformer uses a modified R2 score as base to convert a basic loss into a Healthscore between 0
    and 1.

    Args:
        model (BaseModel): The module to wrap and from which the loss is used to calculate a score.
    """

    def __init__(self, model: BaseModel, remove_noise: bool = False) -> None:
        super().__init__()

        self._model = model
        self._model.eval()
        self._factor = 1

        self._remove_noise = remove_noise
        self._param_noise: Parameter | None = None
        self._param_s_total: Parameter | None = None
        self._param_weight: Parameter | None = None

        self._trained: bool = False

    @property
    def device(self):
        return self._model.device

    @property
    def sub_model(self) -> nn.Module:
        return self._model

    def _iterate(self, dataset: Dataset, loader_args: dict, verbose: bool) -> tuple[list, list]:
        loader_args["batch_size"] = 256
        loader_args["num_workers"] = 6
        loader = DataLoader(dataset, **loader_args)

        data, pred_data = [], []
        for x, y in tqdm(loader, desc="Normalize Data", disable=not verbose, total=len(loader)):
            x = x.to(self._model.device)
            y = y.to(self._model.device)
            y_hat, y = self._model.predict_step(x, y)

            y = y.view(y.size(0) * y.size(1), -1)
            data += y.cpu().detach().tolist()

            y_hat = y_hat.view(y_hat.size(0) * y_hat.size(1), -1)
            pred_data += y_hat.cpu().detach().tolist()

        return data, pred_data

    @torch.no_grad()
    def train_wrapper(self, data: DataModule, reduced: bool = False, verbose: bool = True) -> None:
        logger.debug("Initialize R2 calibration (chained dataset? %s)", "no" if reduced else "yes")

        # load (and combine) the dataset to sample from
        logger.debug("Normalizing the training dataset using the model")
        d = self._iterate(data.trainset, data.loader_args, verbose)[0]
        if not reduced:
            logger.debug("Normalizing the validation dataset using the model")
            _d, _p = self._iterate(data.valset, data.loader_args, verbose)
            d += _d

        # clean up vars and create numpy arrays
        data, d = np.array(d), None
        logger.debug("Calculating parameters on data (shape=%s)", data.shape)

        weight = 1 / (self._factor * data.std(axis=0))
        weight[weight < 1] = self._factor
        self._param_weight = Parameter(Tensor(weight))
        logger.debug("Add weights (min=%s, max=%s)", weight.min(), weight.max())

        s_total = ((data - data.mean(axis=0)) ** 2).mean(axis=0)
        self._param_s_total = Parameter(Tensor(s_total))

        # learn the noise floor to remove it later on
        self._param_noise = Parameter(Tensor(np.zeros_like(s_total)))
        if self._remove_noise and not reduced:
            logger.debug("Train noise floor on validation set")
            noise = np.pow(np.array(_d) - np.array(_p), 2).mean(axis=0)

            logger.debug("Found noise floor: %s", noise)
            self._param_noise = Parameter(Tensor(noise))

        self._trained = True

    @torch.no_grad()
    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, ...]:
        # get the prediction of 'x' using the model
        x = x.to(self._model.device)
        y = y.to(self._model.device)

        # normalize will be handled by the model itself
        y_hat, y = self._model.predict_step(x, y)
        if not self._trained:
            return y_hat, y

        # calculate the score following the R2 formula (note that s_total was modified)
        # also, the noise floor (trained loss) is subtracted from s_res
        s_res = torch.mean(torch.pow(y - y_hat, 2), dim=1).unsqueeze(1)
        s_res = torch.clip(s_res - self._param_noise, 0, None)
        score = 1 - torch.div(s_res, self._param_s_total * self._param_weight)

        _, y_hat = self._model.normalizer.reverse(x=x, y=y_hat)
        return torch.clip(score, 0, 1), y_hat

    def predict_step(self, x: Tensor, y: Tensor) -> tuple[Tensor, ...]:
        # the wrapper must be trained to transfer the prediction error to a score
        if not self._trained:
            y_hat, y = self._model.predict_step(x, y)
            return y_hat, y

        return self.forward(x, y)
