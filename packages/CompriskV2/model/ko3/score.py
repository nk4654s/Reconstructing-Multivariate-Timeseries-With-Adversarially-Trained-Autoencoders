import torch
from torch import Tensor, zeros, zeros_like
from torch.nn import Parameter, Module


class ScoreModule(Module):
    """This module calculates the Healthscore based on the R2 score.
    Integrate this module as follows:

    - `on_training_start()` -> `self.setup()`
    - `training_step()` -> `self.record()`
    - `on_validation_end()` -> `self.finalize()`
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self._data: list | None = None

        self._weight = Parameter(zeros(dim))
        self._total = Parameter(zeros(dim))

    @property
    def ready(self) -> bool:
        return not isinstance(self._data, list)

    def setup(self) -> None:
        """Before recording data, make sure to call this method in
        the module's `on_training_start()` method.
        """
        self._data = []

    def record(self, data: Tensor) -> None:
        if not self.ready:
            temp = data.view(data.size(0) * data.size(1), -1)
            self._data += temp.cpu().detach().tolist()

    def finalize(self, device) -> None:
        if self.ready:
            return None

        self._data = Tensor(self._data).to(device)
        weight = 1 / self._data.std(dim=0)
        weight[weight < 1] = 1
        self._weight = Parameter(weight)

        total = ((self._data - self._data.mean(dim=0)) ** 2).mean(dim=0)
        self._total = Parameter(total)
        self._data = None

    @torch.no_grad()
    def forward(self, y: Tensor, y_hat: Tensor) -> Tensor:
        if not self.ready:
            return zeros_like(y)

        res = torch.pow(y - y_hat, 2)
        res = 1 - torch.div(res, self._total * self._weight)
        return torch.clip(res, 0, 1)
