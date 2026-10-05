from pathlib import Path

import torchvision.transforms as T

from src.experiments.resnet18_experiment import ResNet18Experiment
from src.experiments.swin_transformer_tiny_experiment import (
	SwinTransformerTinyExperiment,
)
from src.experiments.cnn_experiment import CNNExperiment
from src.models import ResNet18, SimpleCNN, SwinTransformerTiny


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
# A 1660 Ti has 6 GB VRAM; keep this fixed for every experiment.
BATCH_SIZE = 4
NUM_WORKERS = 4
PREFETCH_FACTOR = 1
LEARNING_RATE = 1e-4
ACCURACY_TOLERANCE_PERCENT = 10.0
IMAGE_SIZE = 244
CACHE_DIR = Path("data/cache/nutrition5k")
AUGMENTATION_FACTORY = build_train_augmentation
# Five convolutional layers with width 85 give SimpleCNN a parameter count
# close to ResNet18 (11.07M vs 11.18M parameters).
NUM_LAYERS = 5
WIDTH = 85
SEED = 42
EARLY_STOPPING_PATIENCE = 8
GRADIENT_CLIP_NORM = 1.0
GRADIENT_ACCUMULATION_STEPS = 4
USE_AMP = True
RESUME_FROM = None# Example: Path("checkpoints/simple_cnn/2026.09.30_22-21-48/single_split/simple_cnn_best.pt")


def main() -> None:
	common_options = {
		"epochs": EPOCHS,
		"folds": FOLDS,
		"batch_size": BATCH_SIZE,
		"num_workers": NUM_WORKERS,
		"prefetch_factor": PREFETCH_FACTOR,
		"learning_rate": LEARNING_RATE,
		"accuracy_tolerance_percent": ACCURACY_TOLERANCE_PERCENT,
		"image_size": IMAGE_SIZE,
		"cache_dir": CACHE_DIR,
		"augmentation_factory": AUGMENTATION_FACTORY,
		"seed": SEED,
		"early_stopping_patience": EARLY_STOPPING_PATIENCE,
		"gradient_clip_norm": GRADIENT_CLIP_NORM,
		"gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
		"use_amp": USE_AMP,
	}

	experiment_factories = [
		(
			"simple_cnn",
			lambda: SimpleCNN(num_layers=NUM_LAYERS, width=WIDTH),
			lambda: CNNExperiment(
				**common_options,
				num_layers=NUM_LAYERS,
				width=WIDTH,
				resume_from=RESUME_FROM,
			),
		),
		(
			"resnet18",
			ResNet18,
			lambda: ResNet18Experiment(**common_options, resume_from=RESUME_FROM),
		),
		(
			"swin_transformer_tiny",
			SwinTransformerTiny,
			lambda: SwinTransformerTinyExperiment(
				**common_options,
				resume_from=RESUME_FROM,
			),
		),
	]

	for model_name, model_factory, experiment_factory in experiment_factories:
		model = model_factory()
		parameter_count = sum(parameter.numel() for parameter in model.parameters())
		print(f"{model_name}: {parameter_count:,} parameters")
		del model
		print(f"Starting {model_name} experiment...")
		experiment = experiment_factory()
		experiment.run()


if __name__ == "__main__":
	main()
