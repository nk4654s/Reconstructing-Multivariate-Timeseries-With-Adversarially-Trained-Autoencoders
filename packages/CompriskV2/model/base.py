from lightning import LightningModule
from torch import Tensor

from CompriskV2.model.parser import get_loss_by_name, get_optimizer_by_name


def r2_score(i, t):
    mean = t.mean(axis=0)
    ss_regression = ((i - mean) ** 2).sum(axis=0)
    ss_total = ((t - mean) ** 2).sum(axis=0)
    r2_score = ss_regression / ss_total

    return (1 / r2_score.clip(1, None)).mean()


class BaseModel(LightningModule):
    """The BaseModel defines hyper parameters and methods used by every sub-class.

    Args:
        optimizer (str): The optimizer to use, must be registered in
            `CompriskV2.model.utils.get_optimizer_by_name`. Defaults to `"AdamW"`.
        loss (str): The loss function to use, must be registered in
            `CompriskV2.model.utils.get_loss_by_name`. Defaults to `"L1Loss"`.
        lr (float): The gradients learning rate, defaults to `lr=1e-3`.
    """

    def __init__(
        self, optimizer: str = "AdamW", lr: float = 1e-3, loss: str = "L1Loss"
    ) -> None:
        super().__init__()

        self._loss = get_loss_by_name(loss)
        self._optim = get_optimizer_by_name(optimizer)

        self._lr = lr

    @property
    def labels(self) -> tuple[str, ...]:
        """This method defines the names of each model output variable. E.g. if the
        model outputs a the normalized input `x`, a prediction `y_hat`,
        then this function should return `("x_norm", "y_hat")`.

        Returns:
            tuple[str, ...]: Model's output variable names.
        """
        raise NotImplementedError()

    @property
    def onnx_name(self) -> str:
        return "model"

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        y, y_pred = self.predict_step(*batch)

        loss = self._loss(y, y_pred)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        y, y_pred = self.predict_step(*batch)

        loss = self._loss(y, y_pred)
        self.log("val/loss", loss)
        self.log("val/r2", r2_score(y_pred, y))
        return loss

    def configure_optimizers(self):
        if isinstance(self._optim, type):
            self._optim = self._optim(self.parameters(), lr=self._lr)

        return self._optim

    def score(self, *args, **kwargs) -> Tensor:
        """This method defines how to calculate the `Healthscore` for the current sample and
        model output. E.g. `MAE` score implementation:

        ```python
        score(self, y, y_pred) -> Tensor:
            return torch.mean(torch.abs(y - y_pred))
        ````

        > **Note:** All operations within this method must be tracable.
        """
        raise NotImplementedError()

    def predict_step(self, x: Tensor, y: Tensor) -> tuple[Tensor, ...]:
        return self(x, y)
