from pathlib import Path

from src.experiments.cnn_experiment import CNNExperiment


# Experiment settings
EPOCHS = 50
FOLDS = None
BATCH_SIZE = 128
NUM_WORKERS = 8
LEARNING_RATE = 1e-3
ACCURACY_TOLERANCE_PERCENT = 10.0
IMAGE_SIZE = 128
CACHE_DIR = Path("data/cache/nutrition5k")
AUGMENTATION_ENABLED = True
NUM_LAYERS = 4
WIDTH = 64


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
		augmentation_enabled=AUGMENTATION_ENABLED,
		num_layers=NUM_LAYERS,
		width=WIDTH,
	).run()


if __name__ == "__main__":
	main()
