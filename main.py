from pathlib import Path

import torchvision.transforms as T

from src.experiments.cnn_experiment import CNNExperiment


def build_train_augmentation() -> T.Compose:
	return T.Compose(
		[
			T.RandomHorizontalFlip(p=0.5),
			T.RandomRotation(degrees=8),
			T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
		]
	)


# Experiment settings
EPOCHS = 50
FOLDS = None
BATCH_SIZE = 64
NUM_WORKERS = 8
LEARNING_RATE = 1e-3
ACCURACY_TOLERANCE_PERCENT = 10.0
IMAGE_SIZE = 384
CACHE_DIR = Path("data/cache/nutrition5k")
AUGMENTATION_FACTORY = build_train_augmentation
NUM_LAYERS = 3
WIDTH = 32
SEED = 42
EARLY_STOPPING_PATIENCE = 8
GRADIENT_CLIP_NORM = 1.0
USE_AMP = True
RESUME_FROM = Path("checkpoints/simple_cnn/2026.09.30_22-21-48/single_split/simple_cnn_best.pt")


def main() -> None:
	CNNExperiment(
		epochs=EPOCHS,
		folds=FOLDS,
		batch_size=BATCH_SIZE,
		num_workers=NUM_WORKERS,
		learning_rate=LEARNING_RATE,
		accuracy_tolerance_percent=ACCURACY_TOLERANCE_PERCENT,
		image_size=IMAGE_SIZE,
		cache_dir=CACHE_DIR,
		augmentation_factory=AUGMENTATION_FACTORY,
		num_layers=NUM_LAYERS,
		width=WIDTH,
		seed=SEED,
		early_stopping_patience=EARLY_STOPPING_PATIENCE,
		gradient_clip_norm=GRADIENT_CLIP_NORM,
		use_amp=USE_AMP,
		resume_from=RESUME_FROM,
	).run()


if __name__ == "__main__":
	main()
