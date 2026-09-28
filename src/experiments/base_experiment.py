from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from torch import nn
from torch.optim import Optimizer

from src.models import BaseModel
from src.training import test_model, train_model


@dataclass
class FoldSetup:
	model: BaseModel
	train_loader: Iterable[Mapping[str, Any]]
	validation_loader: Iterable[Mapping[str, Any]]
	optimizer: Optimizer
	model_config: Mapping[str, Any]


class BaseExperiment(ABC):
	def __init__(
		self,
		*,
		epochs: int,
		folds: int | None,
		accuracy_tolerance_percent: float,
		augmentation_factory: Callable[[], Any] | None,
		checkpoint_root: Path,
		model_name: str,
		test_loader: Iterable[Mapping[str, Any]],
		test_dish_count: int,
		target_names: list[str],
		criterion: nn.Module,
		device: str,
	) -> None:
		self.epochs = epochs
		self.folds = folds
		self.accuracy_tolerance_percent = accuracy_tolerance_percent
		self.train_augmentation = (
			augmentation_factory() if augmentation_factory is not None else None
		)
		self.checkpoint_root = checkpoint_root
		self.model_name = model_name
		self.test_loader = test_loader
		self.test_dish_count = test_dish_count
		self.target_names = target_names
		self.criterion = criterion
		self.device = device

	@abstractmethod
	def _get_splits(self) -> Iterable[tuple[list[str], list[str]]]:
		"""Return training and validation IDs for each fold."""
		...

	@abstractmethod
	def _create_fold(
		self,
		train_ids: list[str],
		validation_ids: list[str],
		run_label: str,
	) -> FoldSetup:
		"""Build a fold's data loaders, model, optimizer, and configuration."""
		...

	def run(self) -> None:
		for fold_index, (train_ids, validation_ids) in enumerate(
			self._get_splits(), start=1
		):
			run_label = self._run_label(fold_index)
			self._run_fold(fold_index, train_ids, validation_ids, run_label)

	def _run_label(self, fold_index: int) -> str:
		if self.folds is None:
			return "Single split"
		return f"Fold {fold_index}/{self.folds}"

	def _checkpoint_dir_for_fold(self, fold_index: int) -> Path:
		checkpoint_name = (
			f"fold_{fold_index}"
			if self.folds is not None
			else "single_split"
		)
		return self.checkpoint_root / checkpoint_name

	def _run_fold(
		self,
		fold_index: int,
		train_ids: list[str],
		validation_ids: list[str],
		run_label: str,
	) -> None:
		print(
			f"Training {run_label} with {len(train_ids)} train IDs and "
			f"{len(validation_ids)} validation IDs."
		)
		fold = self._create_fold(train_ids, validation_ids, run_label)
		checkpoint_dir = self._checkpoint_dir_for_fold(fold_index)
		checkpoint_dir.mkdir(parents=True, exist_ok=True)
		latest_checkpoint_path = checkpoint_dir / f"{self.model_name}_latest.pt"
		best_checkpoint_path = checkpoint_dir / f"{self.model_name}_best.pt"

		print(
			f"Training start for {run_label} and saving checkpoints "
			f"to {checkpoint_dir}..."
		)
		train_model(
			model=fold.model,
			train_loader=fold.train_loader,
			optimizer=fold.optimizer,
			criterion=self.criterion,
			epochs=self.epochs,
			validation_loader=fold.validation_loader,
			last_checkpoint_path=latest_checkpoint_path,
			best_checkpoint_path=best_checkpoint_path,
			model_config=fold.model_config,
			progress_label=run_label,
			device=self.device,
		)
		self._evaluate_and_save_metrics(
			fold.model,
			best_checkpoint_path,
			fold.model_config,
			checkpoint_dir,
			run_label,
		)

	def _evaluate_and_save_metrics(
		self,
		model: BaseModel,
		checkpoint_path: Path,
		model_config: Mapping[str, Any],
		checkpoint_dir: Path,
		run_label: str,
	) -> None:
		model.load_checkpoint(
			checkpoint_path,
			map_location=self.device,
			expected_model_config=model_config,
		)
		metrics = test_model(
			model,
			self.test_loader,
			self.criterion,
			device=self.device,
			progress_label=f"{run_label} test",
			target_names=self.target_names,
			accuracy_tolerance_percent=self.accuracy_tolerance_percent,
		)
		metrics_payload = {
			"run_label": run_label,
			"model_config": dict(model_config),
			"accuracy_tolerance_percent": self.accuracy_tolerance_percent,
			"test_dish_count": self.test_dish_count,
			**metrics,
		}
		metrics_path = checkpoint_dir / "test_metrics.json"
		with metrics_path.open("w", encoding="utf-8") as metrics_file:
			json.dump(metrics_payload, metrics_file, indent=2, sort_keys=True)
			metrics_file.write("\n")

		print(f"Saved test metrics to {metrics_path}")
		print(f"{run_label} test MSE: {metrics['loss']:.4f}")
		print(f"{run_label} test MAE: {metrics['mae']}")
		print(f"{run_label} test MAPE (%): {metrics['mape_percent']}")
		print(f"{run_label} test R2: {metrics['r2']}")
		print(
			f"{run_label} test Acc@{self.accuracy_tolerance_percent:g}% (%): "
			f"{metrics['acc_at_k_percent']['values_percent']}"
		)
		print(
			"Percentage-metric eligible samples: "
			f"{metrics['percentage_metric_sample_count']}"
		)