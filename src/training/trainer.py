from collections.abc import Callable, Iterable, Mapping
import os
from pathlib import Path
import sys
import time
import math
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


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _write_progress(
    label: str,
    batch_index: int,
    total_batches: Optional[int],
    running_loss: float,
    elapsed_seconds: float,
) -> None:
    if total_batches:
        fraction = min(batch_index / total_batches, 1.0)
        filled = int(24 * fraction)
        bar = "=" * filled + "." * (24 - filled)
        progress = f"[{bar}] {batch_index}/{total_batches} {fraction:.0%}"
    else:
        progress = f"batch {batch_index}"
    elapsed = _format_elapsed(elapsed_seconds)
    sys.stdout.write(f"\r{label} {progress} loss={running_loss:.4f} elapsed={elapsed}")
    sys.stdout.flush()


def _predict_batch(
    model: BaseModel,
    raw_batch: Mapping[str, Any],
    device: torch.device,
    target_key: str,
    target_mean: Optional[Tensor] = None,
    target_std: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    if target_key not in raw_batch:
        raise KeyError(f"Batch is missing target field {target_key!r}.")

    batch = _move_to_device(raw_batch, device)
    targets = batch[target_key]
    if not isinstance(targets, Tensor):
        raise TypeError(f"Batch field {target_key!r} must be a tensor.")
    if targets.ndim == 0:
        raise ValueError("The target tensor must include a batch dimension.")
    targets = _standardize_targets(targets, target_mean, target_std)

    model_inputs = {key: value for key, value in batch.items() if key != target_key}
    predictions = model(model_inputs)
    if not isinstance(predictions, Tensor):
        raise TypeError("The model must return a tensor of predictions.")
    if predictions.shape != targets.shape:
        raise ValueError(
            "Model predictions and targets must have matching shapes; "
            f"got {tuple(predictions.shape)} and {tuple(targets.shape)}."
        )

    return predictions, targets


def _standardize_targets(
    targets: Tensor,
    target_mean: Optional[Tensor],
    target_std: Optional[Tensor],
) -> Tensor:
    if target_mean is None and target_std is None:
        return targets
    if target_mean is None or target_std is None:
        raise ValueError("target_mean and target_std must be provided together.")

    mean = torch.as_tensor(target_mean, device=targets.device, dtype=targets.dtype)
    standard_deviation = torch.as_tensor(
        target_std, device=targets.device, dtype=targets.dtype
    )
    if (
        mean.ndim != 1
        or standard_deviation.shape != mean.shape
        or mean.numel() != targets.shape[-1]
    ):
        raise ValueError("Target statistics must match the target output dimension.")
    if (
        not torch.isfinite(mean).all()
        or not torch.isfinite(standard_deviation).all()
        or (standard_deviation <= 0).any()
    ):
        raise ValueError("Target statistics must be finite with positive standard deviations.")
    return (targets - mean) / standard_deviation


def _restore_target_scale(
    values: Tensor,
    target_mean: Optional[Tensor],
    target_std: Optional[Tensor],
) -> Tensor:
    if target_mean is None and target_std is None:
        return values
    if target_mean is None or target_std is None:
        raise ValueError("target_mean and target_std must be provided together.")

    mean = torch.as_tensor(target_mean, device=values.device, dtype=values.dtype)
    standard_deviation = torch.as_tensor(
        target_std, device=values.device, dtype=values.dtype
    )
    if (
        mean.ndim != 1
        or standard_deviation.shape != mean.shape
        or mean.numel() != values.shape[-1]
    ):
        raise ValueError("Target statistics must match the target output dimension.")
    return values * standard_deviation + mean


def _compute_batch_loss(
    model: BaseModel,
    raw_batch: Mapping[str, Any],
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
    target_key: str,
    target_mean: Optional[Tensor] = None,
    target_std: Optional[Tensor] = None,
) -> tuple[Tensor, int]:
    predictions, targets = _predict_batch(
        model, raw_batch, device, target_key, target_mean, target_std
    )
    loss = criterion(predictions, targets)
    if not isinstance(loss, Tensor) or loss.numel() != 1:
        raise ValueError("The criterion must return a scalar tensor.")
    return loss, targets.shape[0]


def _run_epoch(
    model: BaseModel,
    data_loader: Iterable[Mapping[str, Any]],
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
    target_key: str,
    optimizer: Optional[Optimizer] = None,
    progress_label: Optional[str] = None,
    target_mean: Optional[Tensor] = None,
    target_std: Optional[Tensor] = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    phase_started_at = time.perf_counter()
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
            loss, batch_size = _compute_batch_loss(
                model,
                raw_batch,
                criterion,
                device,
                target_key,
                target_mean,
                target_std,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.detach().item() * batch_size
            sample_count += batch_size
            if progress_label:
                _write_progress(
                    progress_label,
                    batch_count,
                    total_batches,
                    total_loss / sample_count,
                    time.perf_counter() - phase_started_at,
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
    criterion: Callable[[Tensor, Tensor], Tensor],
    epochs: int,
    validation_loader: Optional[Iterable[Mapping[str, Any]]] = None,
    device: Optional[Union[str, torch.device]] = None,
    target_key: str = "targets", # key in loader to get truth
    last_checkpoint_path: Optional[Union[str, Path]] = None,
    best_checkpoint_path: Optional[Union[str, Path]] = None,
    resume_from: Optional[Union[str, Path]] = None,
    model_config: Optional[Mapping[str, Any]] = None,
    progress_label: Optional[str] = None,
    target_mean: Optional[Tensor] = None,
    target_std: Optional[Tensor] = None,
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
    training_started_at = time.perf_counter()
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
        print(f"Resuming training from {resume_from} at epoch {start_epoch + 1}.")

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
                target_mean=target_mean,
                target_std=target_std,
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
                target_mean=target_mean,
                target_std=target_std,
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
        elapsed = _format_elapsed(time.perf_counter() - training_started_at)
        print(
            f"{label}epoch {epoch}/{start_epoch + epochs}: {stats}, "
            f"elapsed={elapsed}",
            flush=True,
        )
        if os.path.exists("./stop_training.txt"):
            print("Stop training file detected after validation. Exiting training loop.")
            Path("./stop_training.txt").unlink(missing_ok=True)
            break

    return history

def test_model(
    model: BaseModel,
    test_loader: Iterable[Mapping[str, Any]],
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: Optional[Union[str, torch.device]] = None,
    target_key: str = "targets",
    progress_label: Optional[str] = "Test",
    target_names: Optional[list[str]] = None,
    accuracy_tolerance_percent: float = 10.0,
    target_mean: Optional[Tensor] = None,
    target_std: Optional[Tensor] = None,
) -> dict[str, Any]:
    """Evaluate regression metrics with bounded memory, one batch at a time.

    MAPE and Acc@K% exclude zero-valued targets because relative error is
    undefined for zero. R2 is reported per output; constant targets yield None.
    """
    if not target_key:
        raise ValueError("target_key must not be empty.")
    if not math.isfinite(accuracy_tolerance_percent) or accuracy_tolerance_percent <= 0:
        raise ValueError("accuracy_tolerance_percent must be a positive finite number.")
    if target_names is not None and len(set(target_names)) != len(target_names):
        raise ValueError("target_names must be unique.")

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    selected_device = torch.device(device if device is not None else default_device)
    model.to(selected_device)
    model.eval()

    sample_count = 0
    batch_count = 0
    total_loss = 0.0
    totals = None
    percentage_counts = None
    accuracy_counts = None
    try:
        total_batches = len(test_loader)
    except (TypeError, NotImplementedError):
        total_batches = None
    evaluation_started_at = time.perf_counter()

    with torch.no_grad():
        for raw_batch in test_loader:
            batch_count += 1
            if not isinstance(raw_batch, Mapping):
                raise TypeError("Each data-loader batch must be a mapping.")
            predictions, targets = _predict_batch(
                model,
                raw_batch,
                selected_device,
                target_key,
                target_mean,
                target_std,
            )
            raw_targets = raw_batch[target_key]
            if not isinstance(raw_targets, Tensor):
                raise TypeError(f"Batch field {target_key!r} must be a tensor.")
            loss = criterion(predictions, targets)
            if not isinstance(loss, Tensor) or loss.numel() != 1:
                raise ValueError("The criterion must return a scalar tensor.")

            predictions_cpu = predictions.detach().to(device="cpu", dtype=torch.float64)
            targets_cpu = targets.detach().to(device="cpu", dtype=torch.float64)
            raw_targets_cpu = raw_targets.detach().to(device="cpu", dtype=torch.float64)
            if predictions_cpu.ndim == 1:
                predictions_cpu = predictions_cpu.unsqueeze(1)
                targets_cpu = targets_cpu.unsqueeze(1)
                raw_targets_cpu = raw_targets_cpu.unsqueeze(1)
            predictions_cpu = _restore_target_scale(
                predictions_cpu, target_mean, target_std
            )
            targets_cpu = _restore_target_scale(targets_cpu, target_mean, target_std)
            if not torch.isfinite(predictions_cpu).all() or not torch.isfinite(targets_cpu).all():
                raise ValueError("Predictions and targets must contain only finite values.")

            if totals is None:
                output_count = predictions_cpu.shape[1]
                if target_names is not None and len(target_names) != output_count:
                    raise ValueError(
                        f"Expected {output_count} target names, got {len(target_names)}."
                    )
                totals = {
                    "absolute_error": torch.zeros(output_count, dtype=torch.float64),
                    "squared_error": torch.zeros(output_count, dtype=torch.float64),
                    "percentage_error": torch.zeros(output_count, dtype=torch.float64),
                    "target_sum": torch.zeros(output_count, dtype=torch.float64),
                    "target_squared_sum": torch.zeros(output_count, dtype=torch.float64),
                }
                percentage_counts = torch.zeros(output_count, dtype=torch.int64)
                accuracy_counts = torch.zeros(output_count, dtype=torch.int64)

            errors = predictions_cpu - targets_cpu
            absolute_errors = errors.abs()
            nonzero_targets = raw_targets_cpu != 0
            relative_errors = torch.zeros_like(absolute_errors)
            relative_errors[nonzero_targets] = (
                absolute_errors[nonzero_targets] / targets_cpu.abs()[nonzero_targets]
            )

            totals["absolute_error"] += absolute_errors.sum(dim=0)
            totals["squared_error"] += errors.square().sum(dim=0)
            totals["percentage_error"] += relative_errors.sum(dim=0)
            totals["target_sum"] += targets_cpu.sum(dim=0)
            totals["target_squared_sum"] += targets_cpu.square().sum(dim=0)
            percentage_counts += nonzero_targets.sum(dim=0)
            accuracy_counts += (
                (relative_errors <= accuracy_tolerance_percent / 100)
                & nonzero_targets
            ).sum(dim=0)

            batch_size = targets_cpu.shape[0]
            sample_count += batch_size
            total_loss += loss.detach().item() * batch_size
            if progress_label:
                _write_progress(
                    progress_label,
                    batch_count,
                    total_batches,
                    total_loss / sample_count,
                    time.perf_counter() - evaluation_started_at,
                )

    if sample_count == 0 or totals is None:
        raise ValueError("The data loader produced no samples.")
    if progress_label:
        sys.stdout.write("\n")
        sys.stdout.flush()

    names = target_names or [f"target_{index}" for index in range(len(totals["absolute_error"]))]
    r2_values = {}
    for index, name in enumerate(names):
        total_sum_squares = (
            totals["target_squared_sum"][index]
            - totals["target_sum"][index].square() / sample_count
        ).item()
        total_sum_squares = max(0.0, total_sum_squares)
        r2_values[name] = (
            1.0 - totals["squared_error"][index].item() / total_sum_squares
            if total_sum_squares > 0
            else None
        )

    mape_values = {}
    accuracy_values = {}
    valid_percentage_counts = {}
    for index, name in enumerate(names):
        valid_count = int(percentage_counts[index].item())
        valid_percentage_counts[name] = valid_count
        if valid_count:
            mape_values[name] = (
                100.0 * totals["percentage_error"][index].item() / valid_count
            )
            accuracy_values[name] = 100.0 * accuracy_counts[index].item() / valid_count
        else:
            mape_values[name] = None
            accuracy_values[name] = None

    return {
        "loss": total_loss / sample_count,
        "sample_count": sample_count,
        "mae": {
            name: totals["absolute_error"][index].item() / sample_count
            for index, name in enumerate(names)
        },
        "mape_percent": mape_values,
        "r2": r2_values,
        "acc_at_k_percent": {
            "k": accuracy_tolerance_percent,
            "values_percent": accuracy_values,
        },
        "percentage_metric_sample_count": valid_percentage_counts,
    }
    