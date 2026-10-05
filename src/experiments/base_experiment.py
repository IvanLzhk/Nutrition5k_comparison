from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import random
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision.transforms as T

from setup import (
	BATCH_SIZE,
	IMAGERY_ROOT,
	METADATA_PATH,
	NUM_WORKERS,
	TEST_IDS_PATH,
	TRAIN_IDS_PATH,
)
from src.dataset.Nutrition5kDataset import Nutrition5kDataset
from src.models import BaseModel
from src.training import get_kfold_splits, test_model, train_model


@dataclass
class FoldSetup:
	model: BaseModel
	train_loader: Iterable[Mapping[str, Any]]
	validation_loader: Iterable[Mapping[str, Any]]
	optimizer: Optimizer
	scheduler: LRScheduler
	model_config: Mapping[str, Any]
	target_mean: Tensor
	target_std: Tensor


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
		target_names: list[str],
		criterion: nn.Module,
		device: str,
		batch_size: int = BATCH_SIZE,
		num_workers: int = NUM_WORKERS,
		prefetch_factor: int = 1,
		learning_rate: float = 1e-3,
		image_size: int = 384,
		cache_dir: Path | None = Path("data/cache/nutrition5k"),
		collate_fn: Callable | None = None,
		seed: int = 42,
		early_stopping_patience: int | None = None,
		gradient_clip_norm: float | None = None,
		gradient_accumulation_steps: int = 1,
		use_amp: bool = False,
		resume_from: Path | None = None,
	) -> None:
		if seed < 0:
			raise ValueError("seed must be non-negative.")
		self.epochs = epochs
		self.folds = folds
		self.accuracy_tolerance_percent = accuracy_tolerance_percent
		self.batch_size = batch_size
		self.num_workers = num_workers
		if prefetch_factor < 1:
			raise ValueError("prefetch_factor must be positive.")
		self.prefetch_factor = prefetch_factor
		self.train_augmentation = (
			augmentation_factory() if augmentation_factory is not None else None
		)
		self.learning_rate = learning_rate
		self.image_size = image_size
		self.cache_dir = cache_dir
		self.metadata_path = Path(METADATA_PATH)
		self.imagery_root = Path(IMAGERY_ROOT)
		self.transform = T.Compose(
			[
				T.Resize((image_size, image_size)),
				T.ToTensor(),
			]
		)
		self.post_transform = T.Normalize(
			[0.485, 0.456, 0.406],
			[0.229, 0.224, 0.225],
		)
		self.test_ids = self._read_ids(Path(TEST_IDS_PATH))
		self.train_ids = self._read_ids(Path(TRAIN_IDS_PATH))
		self._validate_split_ids()
		torch.manual_seed(seed)
		random.seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
		self.checkpoint_root = checkpoint_root
		self.model_name = model_name
		test_dataset = self._create_dataset(self.test_ids)
		if not test_dataset:
			raise ValueError("The configured test IDs produced an empty test dataset.")
		if {entry["dish_id"] for entry in test_dataset.entries} != set(self.test_ids):
			raise ValueError(
				"Configured test IDs do not match usable dataset entries."
			)
		self.test_dish_count = len(self.test_ids)
		self.target_names = target_names
		self.criterion = criterion
		self.device = device
		self.collate_fn = collate_fn
		self.seed = seed
		self.early_stopping_patience = early_stopping_patience
		self.gradient_clip_norm = gradient_clip_norm
		if gradient_accumulation_steps < 1:
			raise ValueError("gradient_accumulation_steps must be positive.")
		self.gradient_accumulation_steps = gradient_accumulation_steps
		self.use_amp = use_amp and device == "cuda"
		self.loader_generator = torch.Generator()
		self.loader_generator.manual_seed(seed)
		self.resume_from = resume_from
		self.test_loader = self._create_loader(test_dataset)

	@staticmethod
	def _read_ids(path: Path) -> list[str]:
		ids = [
			line.strip()
			for line in path.read_text(encoding="utf-8-sig").splitlines()
			if line.strip()
		]
		if len(ids) != len(set(ids)):
			raise ValueError(f"Duplicate dish IDs found in {path}.")
		return ids

	def _validate_split_ids(self) -> None:
		overlap = set(self.train_ids) & set(self.test_ids)
		if overlap:
			raise ValueError(
				f"Train/test dish ID overlap detected: {sorted(overlap)[:5]}"
			)

	def _create_dataset(
		self, dish_ids: list[str], *, training: bool = False
	) -> Nutrition5kDataset:
		return Nutrition5kDataset(
			str(self.metadata_path),
			str(self.imagery_root),
			transform=self.transform,
			dish_ids=dish_ids,
			image_level=True,
			cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
			augmentation=self.train_augmentation if training else None,
			post_transform=getattr(self, "post_transform", None),
		)

	def _create_loader(
		self,
		dataset: Dataset,
		sampler: Sampler | None = None,
	) -> DataLoader:
		loader_options = {
			"batch_size": self.batch_size,
			"num_workers": self.num_workers,
			"pin_memory": getattr(self, "device", "cpu") == "cuda",
			"persistent_workers": self.num_workers > 0,
			"sampler": sampler,
			"collate_fn": getattr(self, "collate_fn", None),
			"worker_init_fn": self._seed_worker,
			"generator": getattr(self, "loader_generator", None),
		}
		if self.num_workers > 0:
			loader_options["prefetch_factor"] = getattr(self, "prefetch_factor", 1)
		return DataLoader(
			dataset,
			**loader_options,
		)

	@staticmethod
	def _seed_worker(worker_id: int) -> None:
		worker_seed = torch.initial_seed() % (2**32)
		random.seed(worker_seed)

	def _get_splits(self) -> Iterable[tuple[list[str], list[str]]]:
		"""Return training and validation IDs for each fold."""
		n_splits = self.folds if self.folds is not None else 5
		splits = get_kfold_splits(
			n_splits=n_splits,
			shuffle=True,
			random_state=42,
			train_ids_path=Path(TRAIN_IDS_PATH),
		)
		if self.folds is None:
			return [next(splits)]
		return list(splits)

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
		# Each fold is one train/validation split. We train a fresh model on
		# that split, then evaluate it before moving to the next fold.
		for fold_index, (train_ids, validation_ids) in enumerate(
			self._get_splits(), start=1
		):
			run_label = self._run_label(fold_index)
			self._run_fold(fold_index, train_ids, validation_ids, run_label)

	def _run_label(self, fold_index: int) -> str:
		# Single split = no cross-validation; otherwise this is the current fold.
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
		# A fold owns one training set, one validation set, and one model run.
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
			scheduler=fold.scheduler,
			criterion=self.criterion,
			epochs=self.epochs,
			validation_loader=fold.validation_loader,
			last_checkpoint_path=latest_checkpoint_path,
			best_checkpoint_path=best_checkpoint_path,
			model_config=fold.model_config,
			progress_label=run_label,
			device=self.device,
			resume_from=self.resume_from,
			target_mean=fold.target_mean,
			target_std=fold.target_std,
			metadata={
				"seed": self.seed,
				"device": self.device,
				"use_amp": self.use_amp,
				"gradient_clip_norm": self.gradient_clip_norm,
				"gradient_accumulation_steps": self.gradient_accumulation_steps,
			},
			early_stopping_patience=self.early_stopping_patience,
			gradient_clip_norm=self.gradient_clip_norm,
			gradient_accumulation_steps=self.gradient_accumulation_steps,
			use_amp=self.use_amp,
		)
		evaluation_checkpoint_path = best_checkpoint_path
		if not (
			evaluation_checkpoint_path.is_file()
			and evaluation_checkpoint_path.with_suffix(".json").is_file()
		):
			if self.resume_from is None:
				raise FileNotFoundError(
					f"Best checkpoint was not created: {best_checkpoint_path}"
				)
			resume_checkpoint_path = Path(self.resume_from)
			resume_metadata_path = resume_checkpoint_path.with_suffix(".json")
			if not resume_checkpoint_path.is_file() or not resume_metadata_path.is_file():
				raise FileNotFoundError(
					f"Resume checkpoint pair was not found: {resume_checkpoint_path}"
				)
			print(
				"No new best checkpoint was created; copying the selected "
				f"resume checkpoint {self.resume_from} into {checkpoint_dir}."
			)
			shutil.copy2(resume_checkpoint_path, best_checkpoint_path)
			shutil.copy2(
				resume_metadata_path, best_checkpoint_path.with_suffix(".json")
			)
		self._evaluate_and_save_metrics(
			fold.model,
			evaluation_checkpoint_path,
			fold.model_config,
			checkpoint_dir,
			run_label,
			fold.target_mean,
			fold.target_std,
		)

	def _evaluate_and_save_metrics(
		self,
		model: BaseModel,
		checkpoint_path: Path,
		model_config: Mapping[str, Any],
		checkpoint_dir: Path,
		run_label: str,
		target_mean: Tensor,
		target_std: Tensor,
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
			target_mean=target_mean,
			target_std=target_std,
		)
		metrics_payload = {
			"run_label": run_label,
			"model_config": dict(model_config),
			"loss_space": "standardized targets",
			"accuracy_tolerance_percent": self.accuracy_tolerance_percent,
			"test_dish_count": self.test_dish_count,
			**metrics,
		}
		metrics_path = checkpoint_dir / "test_metrics.json"
		with metrics_path.open("w", encoding="utf-8") as metrics_file:
			json.dump(metrics_payload, metrics_file, indent=2, sort_keys=True)
			metrics_file.write("\n")

		print(f"Saved test metrics to {metrics_path}")
		print(f"{run_label} test standardized MSE: {metrics['loss']:.4f}")
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