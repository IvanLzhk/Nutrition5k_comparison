import argparse
import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from setup import BATCH_SIZE, IMAGERY_ROOT, METADATA_PATH, MODEL_CHECKPOINT_DIR, NUM_WORKERS, TRAIN_IDS_PATH
from src.dataset.Nutrition5kDataset import Nutrition5kDataset
from src.models import SimpleCNN
from src.training import get_kfold_splits, train_model


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
	loader_options = {
		"batch_size": args.batch_size,
		"num_workers": args.num_workers,
		"pin_memory": True,
	}
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
		print(f"Training start for {run_label} and saving checkpoints to {checkpoint_dir}...")
		train_model(
			model=model,
			train_loader=train_loader,
			optimizer=optimizer,
			criterion=nn.MSELoss(),
			epochs=args.epochs,
			validation_loader=validation_loader,
			last_checkpoint_path=checkpoint_dir / "simple_cnn_latest.pt",
			best_checkpoint_path=checkpoint_dir / "simple_cnn_best.pt",
			model_config=model_config,
			progress_label=run_label,
			device="cuda",
		)


if __name__ == "__main__":
	main()
