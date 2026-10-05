import datetime
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

from src.models import ResNet18
from setup import BATCH_SIZE, MODEL_CHECKPOINT_DIR, NUM_WORKERS
from .base_experiment import BaseExperiment, FoldSetup

class ResNet18Experiment(BaseExperiment):
    TARGET_NAMES = ["calories", "mass_g", "fat_g", "carbs_g", "protein_g"]

    def __init__(
            self,
            *,
            augmentation_factory: Callable[[], Any] | None,
            epochs: int = 50,
            folds: int | None = None,
            batch_size: int = BATCH_SIZE,
            num_workers: int = NUM_WORKERS,
            prefetch_factor: int = 1,
            learning_rate: float = 1e-3,
            accuracy_tolerance_percent: float = 10.0,
            image_size: int = 244,
            cache_dir: Path | None = Path("data/cache/nutrition5k"),
            seed: int = 42,
            early_stopping_patience: int | None = 8,
            gradient_clip_norm: float | None = 1.0,
            gradient_accumulation_steps: int = 1,
            use_amp: bool = True,
            resume_from: Path | None = None,
        ) -> None:
            checkpoint_root = (
                    Path(MODEL_CHECKPOINT_DIR)
                    / "resnet18"
                    / datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
                )
            super().__init__(
                epochs=epochs,
                folds=folds,
                accuracy_tolerance_percent=accuracy_tolerance_percent,
                augmentation_factory=augmentation_factory,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                learning_rate=learning_rate,
                image_size=image_size,
                cache_dir=cache_dir,
                checkpoint_root=checkpoint_root,
                model_name="resnet18",
                target_names=self.TARGET_NAMES,
                criterion=nn.MSELoss(),
                device="cuda" if torch.cuda.is_available() else "cpu",
                collate_fn=None,
                seed=seed,
                early_stopping_patience=early_stopping_patience,
                gradient_clip_norm=gradient_clip_norm,
                gradient_accumulation_steps=gradient_accumulation_steps,
                use_amp=use_amp,
                resume_from=resume_from,
            )

    def _create_fold(
            self,
            train_ids: list[str],
            validation_ids: list[str],
            run_label: str,
    ) -> FoldSetup:
        train_dataset = self._create_dataset(train_ids, training=True)
        validation_dataset = self._create_dataset(validation_ids, image_level=False)
        if not train_dataset or not validation_dataset:
            raise ValueError(
                f"{run_label} has an empty train or validation dataset. "
                "Check the configured training IDs and available imagery."
            )

        train_entry_ids = {entry["dish_id"] for entry in train_dataset.entries}
        validation_entry_ids = {
            entry["dish_id"] for entry in validation_dataset.entries
        }
        if train_entry_ids != set(train_ids):
            raise ValueError(
                f"{run_label} train IDs do not match usable dataset entries."
            )
        if validation_entry_ids != set(validation_ids):
            raise ValueError(
                f"{run_label} validation IDs do not match usable dataset entries."
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
            generator=self.loader_generator,
        )
        train_loader = self._create_loader(train_dataset, sampler=train_sampler)
        validation_loader = self._create_loader(validation_dataset)

        model = ResNet18(output_features=len(self.TARGET_NAMES))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        model_config = {
            "output_features": len(self.TARGET_NAMES),
            "image_size": self.image_size,
            "uses_single_view": True,
            "backbone": "resnet18",
            "target_mean": target_mean.tolist(),
            "target_std": target_std.tolist(),
            "seed": self.seed,
            "early_stopping_patience": self.early_stopping_patience,
            "gradient_clip_norm": self.gradient_clip_norm,
            "use_amp": self.use_amp,
        }
        return FoldSetup(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            target_mean=target_mean,
            target_std=target_std,
        )