from collections.abc import Callable, Iterable, Mapping
from typing import Any, Optional, Union

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from src.models import BaseModel


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _run_epoch(
    model: BaseModel,
    data_loader: Iterable[Mapping[str, Any]],
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
    target_key: str,
    optimizer: Optional[Optimizer] = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0

    with torch.set_grad_enabled(training):
        for raw_batch in data_loader:
            if not isinstance(raw_batch, Mapping):
                raise TypeError("Each data-loader batch must be a mapping.")
            if target_key not in raw_batch:
                raise KeyError(f"Batch is missing target field {target_key!r}.")

            batch = _move_to_device(raw_batch, device)
            targets = batch[target_key]
            if not isinstance(targets, Tensor):
                raise TypeError(f"Batch field {target_key!r} must be a tensor.")
            if targets.ndim == 0:
                raise ValueError("The target tensor must include a batch dimension.")

            model_inputs = {key: value for key, value in batch.items() if key != target_key}
            predictions = model(model_inputs)
            if not isinstance(predictions, Tensor):
                raise TypeError("The model must return a tensor of predictions.")
            if predictions.shape != targets.shape:
                raise ValueError(
                    "Model predictions and targets must have matching shapes; "
                    f"got {tuple(predictions.shape)} and {tuple(targets.shape)}."
                )

            loss = criterion(predictions, targets)
            if not isinstance(loss, Tensor) or loss.numel() != 1:
                raise ValueError("The criterion must return a scalar tensor.")

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = targets.shape[0]
            total_loss += loss.detach().item() * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("The data loader produced no samples.")
    return total_loss / sample_count


def train_model(
    model: BaseModel,
    train_loader: Iterable[Mapping[str, Any]],
    optimizer: Optimizer,
    criterion: Callable[[Tensor, Tensor], Tensor],
    epochs: int,
    validation_loader: Optional[Iterable[Mapping[str, Any]]] = None,
    device: Optional[Union[str, torch.device]] = None,
    target_key: str = "targets",
) -> dict[str, list[float]]:
    """Train a ``BaseModel`` and return sample-weighted losses for each epoch.

    Batches must be mappings containing ``target_key`` and any model inputs.
    The model receives all batch fields except the target. The criterion should
    return a scalar mean loss (for example, ``nn.MSELoss()``).
    """
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if not target_key:
        raise ValueError("target_key must not be empty.")

    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(selected_device)

    history = {"train_loss": []}
    if validation_loader is not None:
        history["val_loss"] = []

    for _ in range(epochs):
        history["train_loss"].append(
            _run_epoch(
                model,
                train_loader,
                criterion,
                selected_device,
                target_key,
                optimizer,
            )
        )
        if validation_loader is not None:
            history["val_loss"].append(
                _run_epoch(
                    model,
                    validation_loader,
                    criterion,
                    selected_device,
                    target_key,
                )
            )

    return history
