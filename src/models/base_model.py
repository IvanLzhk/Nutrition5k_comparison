from abc import ABC, abstractmethod
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Union

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


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

    @staticmethod
    def _checkpoint_class_name(value: Any) -> str:
        model_type = type(value)
        return f"{model_type.__module__}.{model_type.__qualname__}"

    @staticmethod
    def _validate_json_data(name: str, value: Any) -> None:
        try:
            json.dumps(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain only JSON-serializable values.") from error

    def save_checkpoint(
        self,
        path: Union[str, Path],
        *,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        epoch: int = 0,
        history: Optional[Mapping[str, Any]] = None,
        best_val_loss: Optional[float] = None,
        best_epoch: Optional[int] = None,
        model_config: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Save tensor state to ``.pt`` and readable progress to a sibling ``.json``."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")

        checkpoint_history = dict(history) if history is not None else {}
        checkpoint_config = dict(model_config) if model_config is not None else None
        checkpoint_metadata = dict(metadata) if metadata is not None else None
        self._validate_json_data("history", checkpoint_history)
        if checkpoint_config is not None:
            self._validate_json_data("model_config", checkpoint_config)
        if checkpoint_metadata is not None:
            self._validate_json_data("metadata", checkpoint_metadata)

        payload = {
            "format_version": 1,
            "model_class": self._checkpoint_class_name(self),
            "model_state_dict": self.state_dict(),
            "optimizer_class": self._checkpoint_class_name(optimizer) if optimizer else None,
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_class": self._checkpoint_class_name(scheduler) if scheduler else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        }
        metadata_payload = {
            "format_version": 1,
            "model_class": self._checkpoint_class_name(self),
            "model_config": checkpoint_config,
            "epoch": epoch,
            "history": checkpoint_history,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "metadata": checkpoint_metadata,
        }

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = checkpoint_path.with_suffix(".json")
        temporary_paths = []
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=checkpoint_path.parent,
                prefix=f"{metadata_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                metadata_temp_path = Path(temporary_file.name)
                temporary_paths.append(metadata_temp_path)
                json.dump(metadata_payload, temporary_file, indent=2, sort_keys=True)
                temporary_file.write("\n")

            with tempfile.NamedTemporaryFile(
                dir=checkpoint_path.parent,
                prefix=f"{checkpoint_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                checkpoint_temp_path = Path(temporary_file.name)
                temporary_paths.append(checkpoint_temp_path)
            torch.save(payload, checkpoint_temp_path)

            checkpoint_temp_path.replace(checkpoint_path)
            metadata_temp_path.replace(metadata_path)
        finally:
            for temporary_path in temporary_paths:
                temporary_path.unlink(missing_ok=True)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        *,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        map_location: Union[str, torch.device] = "cpu",
        expected_model_config: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Restore model state and progress from a checkpoint pair."""
        checkpoint_path = Path(path)
        metadata_path = checkpoint_path.with_suffix(".json")
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            progress = json.load(metadata_file)
        if not isinstance(progress, Mapping) or progress.get("format_version") != 1:
            raise ValueError("Unsupported or invalid checkpoint metadata.")

        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(checkpoint, Mapping) or checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported or invalid model checkpoint.")

        if progress.get("model_class") != self._checkpoint_class_name(self):
            raise ValueError(
                "Checkpoint model class does not match the provided model instance."
            )

        saved_config = progress.get("model_config")
        if expected_model_config is not None:
            self._validate_json_data("expected_model_config", dict(expected_model_config))
            if saved_config != dict(expected_model_config):
                raise ValueError("Checkpoint model configuration does not match.")

        self.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            if checkpoint.get("optimizer_state_dict") is None:
                raise ValueError("Checkpoint does not contain optimizer state.")
            if checkpoint.get("optimizer_class") != self._checkpoint_class_name(optimizer):
                raise ValueError(
                    "Checkpoint optimizer class does not match the provided optimizer."
                )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            if checkpoint.get("scheduler_state_dict") is None:
                raise ValueError("Checkpoint does not contain scheduler state.")
            if checkpoint.get("scheduler_class") != self._checkpoint_class_name(scheduler):
                raise ValueError(
                    "Checkpoint scheduler class does not match the provided scheduler."
                )
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "epoch": progress.get("epoch", 0),
            "history": progress.get("history", {}),
            "best_val_loss": progress.get("best_val_loss"),
            "best_epoch": progress.get("best_epoch"),
            "model_config": saved_config,
            "metadata": progress.get("metadata"),
        }
