import datetime
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler
import torchvision.transforms as T

from setup import (
	BATCH_SIZE,
	IMAGERY_ROOT,
	METADATA_PATH,
	MODEL_CHECKPOINT_DIR,
	NUM_WORKERS,
	TEST_IDS_PATH,
	TRAIN_IDS_PATH,
)
from src.dataset.Nutrition5kDataset import Nutrition5kDataset
from src.models import SimpleCNN
from src.training import get_kfold_splits

from .base_experiment import BaseExperiment, FoldSetup


class CNNExperiment(BaseExperiment):
	TARGET_NAMES = ["calories", "mass_g", "fat_g", "carbs_g", "protein_g"]

	def __init__(
		self,
		*,
		augmentation_factory: Callable[[], Any] | None,
		epochs: int = 50,
		folds: int | None = None,
		batch_size: int = BATCH_SIZE,
		num_workers: int = NUM_WORKERS,
		learning_rate: float = 1e-3,
		accuracy_tolerance_percent: float = 10.0,
		image_size: int = 128,
		cache_dir: Path | None = Path("data/cache/nutrition5k"),
		num_layers: int = 4,
		width: int = 64,
	) -> None:
		self.learning_rate = learning_rate
		self.image_size = image_size
		self.num_layers = num_layers
		self.width = width
		self.cache_dir = cache_dir
		self.metadata_path = Path(METADATA_PATH)
		self.imagery_root = Path(IMAGERY_ROOT)
		self.transform = T.Compose(
			[T.Resize((image_size, image_size)), T.ToTensor()]
		)
		self.test_ids = self._read_ids(Path(TEST_IDS_PATH))
		test_dataset = self._create_dataset(self.test_ids)
		if not test_dataset:
			raise ValueError("The configured test IDs produced an empty test dataset.")
		checkpoint_root = (
			Path(MODEL_CHECKPOINT_DIR)
			/ "simple_cnn"
			/ datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
		)
		super().__init__(
			epochs=epochs,
			folds=folds,
			accuracy_tolerance_percent=accuracy_tolerance_percent,
			augmentation_factory=augmentation_factory,
			batch_size=batch_size,
			num_workers=num_workers,
			checkpoint_root=checkpoint_root,
			model_name="simple_cnn",
			test_dataset=test_dataset,
			test_dish_count=len(self.test_ids),
			target_names=self.TARGET_NAMES,
			criterion=nn.MSELoss(),
			device="cuda",
		)

	@staticmethod
	def _read_ids(path: Path) -> list[str]:
		return [
			line.strip()
			for line in path.read_text(encoding="utf-8-sig").splitlines()
			if line.strip()
		]

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
		)

	def _get_splits(self) -> list[tuple[list[str], list[str]]]:
		# Build a set of train/validation partitions. Each partition is one fold.
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

	def _create_fold(
		self,
		train_ids: list[str],
		validation_ids: list[str],
		run_label: str,
	) -> FoldSetup:
		# A fold = one model training run on one slice of the dataset.
		train_dataset = self._create_dataset(train_ids, training=True)
		validation_dataset = self._create_dataset(validation_ids)
		if not train_dataset or not validation_dataset:
			raise ValueError(
				f"{run_label} has an empty train or validation dataset. "
				"Check the configured training IDs and available imagery."
			)
		training_targets = torch.tensor(
			[entry["targets"] for entry in train_dataset.entries],
			dtype=torch.float32,
		)
		target_mean = training_targets.mean(dim=0)
		target_std = training_targets.std(dim=0, unbiased=False).clamp_min(1e-6)

		train_sampler = WeightedRandomSampler(
			train_dataset.sample_weights,
			num_samples=len(train_dataset),
			replacement=True,
		)
		train_loader = self._create_loader(train_dataset, sampler=train_sampler)
		validation_loader = self._create_loader(validation_dataset)
		print(
			f"Initializing SimpleCNN with {self.num_layers} layers "
			f"and width {self.width}..."
		)
		model = SimpleCNN(
			output_features=5,
			num_layers=self.num_layers,
			width=self.width,
		)
		optimizer = torch.optim.Adam(
			model.parameters(), lr=self.learning_rate
		)
		model_config = {
			"output_features": 5,
			"image_size": self.image_size,
			"num_layers": self.num_layers,
			"width": self.width,
			"target_mean": target_mean.tolist(),
			"target_std": target_std.tolist(),
		}
		return FoldSetup(
			model=model,
			train_loader=train_loader,
			validation_loader=validation_loader,
			optimizer=optimizer,
			model_config=model_config,
			target_mean=target_mean,
			target_std=target_std,
		)