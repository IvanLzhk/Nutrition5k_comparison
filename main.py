import argparse
import datetime
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from setup import BATCH_SIZE, IMAGERY_ROOT, METADATA_PATH, MODEL_CHECKPOINT_DIR, NUM_WORKERS, TEST_IDS_PATH, TRAIN_IDS_PATH
from src.dataset.Nutrition5kDataset import Nutrition5kDataset
from src.models import SimpleCNN
from src.training import get_kfold_splits, test_model, train_model


def main() -> None:
	parser = argparse.ArgumentParser(description="Train the simple Nutrition5k CNN.")
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument(
		"--folds",
		type=int,
		default=None,
		help="Run full k-fold cross-validation; omit for one 80/20 train/validation split.",
	)
	parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
	parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--accuracy-tolerance-percent", type=float, default=10.0)
	parser.add_argument("--image-size", type=int, default=128)
	parser.add_argument(
		"--num-layers",
		type=int,
		default=3,
		help="Number of convolutional layers.",
	)
	parser.add_argument(
		"--width",
		type=int,
		default=16,
		help="Channels in the first layer; channels double at each layer.",
	)
	args = parser.parse_args()

	transform = T.Compose(
		[T.Resize((args.image_size, args.image_size)), T.ToTensor()]
	)
	metadata_path = Path(METADATA_PATH)
	imagery_root = Path(IMAGERY_ROOT)
	test_ids = [
		line.strip()
		for line in Path(TEST_IDS_PATH).read_text(encoding="utf-8-sig").splitlines()
		if line.strip()
	]
	test_dataset = Nutrition5kDataset(
		str(metadata_path),
		str(imagery_root),
		transform=transform,
		dish_ids=test_ids,
		image_level=True,
	)
	if not test_dataset:
		raise ValueError("The configured test IDs produced an empty test dataset.")
	loader_options = {
		"batch_size": args.batch_size,
		"num_workers": args.num_workers,
		"pin_memory": True,
	}
	test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
	checkpoint_root = (
		Path(MODEL_CHECKPOINT_DIR)
		/ "simple_cnn"
		/ datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
	)
	n_splits = args.folds if args.folds is not None else 5
	fold_splits = get_kfold_splits(
		n_splits=n_splits,
		shuffle=True,
		random_state=42,
		train_ids_path=Path(TRAIN_IDS_PATH),
	)
	if args.folds is None:
		fold_splits = [next(fold_splits)]

	for fold_index, (train_ids, validation_ids) in enumerate(fold_splits, start=1):
		run_label = (
			f"Fold {fold_index}/{args.folds}"
			if args.folds is not None
			else "Single split"
		)
		print(f"Training {run_label} with {len(train_ids)} train IDs and {len(validation_ids)} validation IDs.")
		train_dataset = Nutrition5kDataset(
			str(metadata_path),
			str(imagery_root),
			transform=transform,
			dish_ids=train_ids,
			image_level=True,
		)
		validation_dataset = Nutrition5kDataset(
			str(metadata_path),
			str(imagery_root),
			transform=transform,
			dish_ids=validation_ids,
			image_level=True,
		)
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
		train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_options)
		validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
		print(f"Initializing SimpleCNN with {args.num_layers} layers and width {args.width}...")
		model = SimpleCNN(
			output_features=5,
			num_layers=args.num_layers,
			width=args.width,
		)
		optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
		model_config = {
			"output_features": 5,
			"image_size": args.image_size,
			"num_layers": args.num_layers,
			"width": args.width,
		}
		checkpoint_name = f"fold_{fold_index}" if args.folds is not None else "single_split"
		checkpoint_dir = checkpoint_root / checkpoint_name
		checkpoint_dir.mkdir(parents=True, exist_ok=True)
		best_checkpoint_path = checkpoint_dir / "simple_cnn_best.pt"
		print(f"Training start for {run_label} and saving checkpoints to {checkpoint_dir}...")
		train_model(
			model=model,
			train_loader=train_loader,
			optimizer=optimizer,
			criterion=nn.MSELoss(),
			epochs=args.epochs,
			validation_loader=validation_loader,
			last_checkpoint_path=checkpoint_dir / "simple_cnn_latest.pt",
			best_checkpoint_path=best_checkpoint_path,
			model_config=model_config,
			progress_label=run_label,
			device="cuda",
		)
		model.load_checkpoint(
			best_checkpoint_path,
			map_location="cuda",
			expected_model_config=model_config,
		)
		test_metrics = test_model(
			model,
			test_loader,
			nn.MSELoss(),
			device="cuda",
			progress_label=f"{run_label} test",
			target_names=["calories", "mass_g", "fat_g", "carbs_g", "protein_g"],
			accuracy_tolerance_percent=args.accuracy_tolerance_percent,
		)
		metrics_path = checkpoint_dir / "test_metrics.json"
		metrics_payload = {
			"run_label": run_label,
			"model_config": model_config,
			"accuracy_tolerance_percent": args.accuracy_tolerance_percent,
			"test_dish_count": len(test_ids),
			**test_metrics,
		}
		with metrics_path.open("w", encoding="utf-8") as metrics_file:
			json.dump(metrics_payload, metrics_file, indent=2, sort_keys=True)
			metrics_file.write("\n")
		print(f"Saved test metrics to {metrics_path}")
		print(f"{run_label} test MSE: {test_metrics['loss']:.4f}")
		print(f"{run_label} test MAE: {test_metrics['mae']}")
		print(f"{run_label} test MAPE (%): {test_metrics['mape_percent']}")
		print(f"{run_label} test R2: {test_metrics['r2']}")
		print(
			f"{run_label} test Acc@{args.accuracy_tolerance_percent:g}% (%): "
			f"{test_metrics['acc_at_k_percent']['values_percent']}"
		)
		print(
			f"Percentage-metric eligible samples: "
			f"{test_metrics['percentage_metric_sample_count']}"
		)


if __name__ == "__main__":
	main()
