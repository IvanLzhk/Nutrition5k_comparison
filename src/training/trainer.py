from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
import sys
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


def _write_progress(
    label: str,
    batch_index: int,
    total_batches: Optional[int],
    running_loss: float,
) -> None:
    if total_batches:
        fraction = min(batch_index / total_batches, 1.0)
        filled = int(24 * fraction)
        bar = "=" * filled + "." * (24 - filled)
        progress = f"[{bar}] {batch_index}/{total_batches} {fraction:.0%}"
    else:
        progress = f"batch {batch_index}"
    sys.stdout.write(f"\r{label} {progress} loss={running_loss:.4f}")
    sys.stdout.flush()


def _run_epoch(
    model: BaseModel,
    data_loader: Iterable[Mapping[str, Any]],
    criterion: Callable[[Tensor, Tensor], Tensor], # loss func
    device: torch.device,
    target_key: str,
    optimizer: Optional[Optimizer] = None,
    progress_label: Optional[str] = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    batch_count = 0
    try:
        total_batches = len(data_loader)
    except (TypeError, NotImplementedError):
        total_batches = None

    with torch.set_grad_enabled(training):
        for raw_batch in data_loader:
            batch_count += 1
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
            if progress_label:
                _write_progress(
                    progress_label,
                    batch_count,
                    total_batches,
                    total_loss / sample_count,
                )

    if sample_count == 0:
        raise ValueError("The data loader produced no samples.")
    if progress_label:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return total_loss / sample_count


def train_model(
    model: BaseModel,
    train_loader: Iterable[Mapping[str, Any]],
    optimizer: Optimizer,
    criterion: Callable[[Tensor, Tensor], Tensor], # loss function
    epochs: int,
    validation_loader: Optional[Iterable[Mapping[str, Any]]] = None,
    device: Optional[Union[str, torch.device]] = None,
    target_key: str = "targets", # key in loader to get truth
    last_checkpoint_path: Optional[Union[str, Path]] = None,
    best_checkpoint_path: Optional[Union[str, Path]] = None,
    resume_from: Optional[Union[str, Path]] = None,
    model_config: Optional[Mapping[str, Any]] = None,
    progress_label: Optional[str] = None,
) -> dict[str, list[float]]:
    """Train a ``BaseModel`` on CUDA by default and return sample-weighted losses.

    Batches must be mappings containing ``target_key`` and any model inputs.
    The model receives all batch fields except the target. The criterion should
    return a scalar mean loss (for example, ``nn.MSELoss()``). Pass ``device="cpu"``
    to explicitly train on the CPU.
    """
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if not target_key:
        raise ValueError("target_key must not be empty.")
    if best_checkpoint_path is not None and validation_loader is None:
        raise ValueError("best_checkpoint_path requires a validation_loader.")
    if (
        last_checkpoint_path is not None
        and best_checkpoint_path is not None
        and Path(last_checkpoint_path) == Path(best_checkpoint_path)
    ):
        raise ValueError("Latest and best checkpoints must use different paths.")

    selected_device = torch.device(device if device is not None else "cuda")
    model.to(selected_device)

    start_epoch = 0
    best_val_loss = None
    best_epoch = None
    history = {"train_loss": []}
    if resume_from is not None:
        progress = model.load_checkpoint(
            resume_from,
            optimizer=optimizer,
            map_location=selected_device,
            expected_model_config=model_config,
        )
        start_epoch = progress["epoch"]
        history = progress["history"]
        best_val_loss = progress["best_val_loss"]
        best_epoch = progress["best_epoch"]
        if not isinstance(history, dict) or not isinstance(history.get("train_loss"), list):
            raise ValueError("Checkpoint history must include a train_loss list.")

    if validation_loader is not None:
        history.setdefault("val_loss", [])

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        epoch_label = f"{progress_label}, " if progress_label else ""
        epoch_label += f"epoch {epoch}/{start_epoch + epochs}"
        history["train_loss"].append(
            _run_epoch(
                model,
                train_loader,
                criterion,
                selected_device,
                target_key,
                optimizer,
                progress_label=f"{epoch_label} train",
            )
        )
        if validation_loader is not None:
            val_loss = _run_epoch(
                model,
                validation_loader,
                criterion,
                selected_device,
                target_key,
                progress_label=f"{epoch_label} val",
            )
            history["val_loss"].append(val_loss)
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                if best_checkpoint_path is not None:
                    model.save_checkpoint(
                        best_checkpoint_path,
                        optimizer=optimizer,
                        epoch=epoch,
                        history=history,
                        best_val_loss=best_val_loss,
                        best_epoch=best_epoch,
                        model_config=model_config,
                    )

        if last_checkpoint_path is not None:
            model.save_checkpoint(
                last_checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                model_config=model_config,
            )

        stats = f"train_loss={history['train_loss'][-1]:.4f}"
        if validation_loader is not None:
            stats += f", val_loss={history['val_loss'][-1]:.4f}"
        label = f"{progress_label}, " if progress_label else ""
        print(f"{label}epoch {epoch}/{start_epoch + epochs}: {stats}", flush=True)

    return history