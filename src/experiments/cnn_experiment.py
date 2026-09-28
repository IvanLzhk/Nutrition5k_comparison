import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from setup import (
	IMAGERY_ROOT,
	METADATA_PATH,
	MODEL_CHECKPOINT_DIR,
	TEST_IDS_PATH,
	TRAIN_IDS_PATH,
)
from src.dataset.Nutrition5kDataset import Nutrition5kDataset
from src.models import SimpleCNN
from src.training import get_kfold_splits

from .base_experiment import BaseExperiment, FoldSetup


class CNNExperiment(BaseExperiment):
	TARGET_NAMES = ["calories", "mass_g", "fat_g", "carbs_g", "protein_g"]

	def __init__(self, args: Any) -> None:
		self.args = args
		self.cache_dir = (
			None
			if getattr(args, "no_cache", False)
			else Path(getattr(args, "cache_dir", "data/cache/nutrition5k"))
		)
		self.metadata_path = Path(METADATA_PATH)
		self.imagery_root = Path(IMAGERY_ROOT)
		self.transform = T.Compose(
			[T.Resize((args.image_size, args.image_size)), T.ToTensor()]
		)
		self.test_ids = self._read_ids(Path(TEST_IDS_PATH))
		test_dataset = self._create_dataset(self.test_ids)
		if not test_dataset:
			raise ValueError("The configured test IDs produced an empty test dataset.")
		test_loader = self._create_loader(test_dataset)
		checkpoint_root = (
			Path(MODEL_CHECKPOINT_DIR)
			/ "simple_cnn"
			/ datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
		)
		super().__init__(
			args,
			checkpoint_root=checkpoint_root,
			model_name="simple_cnn",
			test_loader=test_loader,
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

	def _create_loader(
		self,
		dataset: Nutrition5kDataset,
		sampler: Optional[WeightedRandomSampler] = None,
	) -> DataLoader:
		return DataLoader(
			dataset,
			batch_size=self.args.batch_size,
			num_workers=self.args.num_workers,
			pin_memory=True,
			sampler=sampler,
		)

	def _get_splits(self) -> list[tuple[list[str], list[str]]]:
		n_splits = self.args.folds if self.args.folds is not None else 5
		splits = get_kfold_splits(
			n_splits=n_splits,
			shuffle=True,
			random_state=42,
			train_ids_path=Path(TRAIN_IDS_PATH),
		)
		if self.args.folds is None:
			return [next(splits)]
		return list(splits)

	def _create_fold(
		self,
		train_ids: list[str],
		validation_ids: list[str],
		run_label: str,
	) -> FoldSetup:
		train_dataset = self._create_dataset(train_ids, training=True)
		validation_dataset = self._create_dataset(validation_ids)
		if not train_dataset or not validation_dataset:
			raise ValueError(
				f"{run_label} has an empty train or validation dataset. "
				"Check the configured training IDs and available imagery."
			)

		train_sampler = WeightedRandomSampler(
			train_dataset.sample_weights,
			num_samples=len(train_dataset),
			replacement=True,
		)
		train_loader = self._create_loader(train_dataset, sampler=train_sampler)
		validation_loader = self._create_loader(validation_dataset)
		print(
			f"Initializing SimpleCNN with {self.args.num_layers} layers "
			f"and width {self.args.width}..."
		)
		model = SimpleCNN(
			output_features=5,
			num_layers=self.args.num_layers,
			width=self.args.width,
		)
		optimizer = torch.optim.Adam(
			model.parameters(), lr=self.args.learning_rate
		)
		model_config = {
			"output_features": 5,
			"image_size": self.args.image_size,
			"num_layers": self.args.num_layers,
			"width": self.args.width,
		}
		return FoldSetup(
			model=model,
			train_loader=train_loader,
			validation_loader=validation_loader,
			optimizer=optimizer,
			model_config=model_config,
		)