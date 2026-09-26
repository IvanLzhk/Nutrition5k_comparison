from abc import ABC, abstractmethod
from typing import Any, Mapping

from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    """Base class for models trained with the project's batch-based trainer.

    Subclasses implement ``forward`` and return predictions with the same
    shape as the batch target. The trainer passes a mapping of model inputs;
    the target field is excluded.
    """

    @abstractmethod
    def forward(self, batch: Mapping[str, Any]) -> Tensor:
        """Return predictions for a batch of model inputs."""
        raise NotImplementedError
