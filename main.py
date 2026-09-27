import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from setup import BATCH_SIZE, IMAGERY_ROOT, METADATA_PATH, MODEL_CHECKPOINT_DIR, NUM_WORKERS, TEST_IDS_PATH, TRAIN_IDS_PATH
from src.dataset.Nutrition5kDataset import Nutrition5kDataset, collate_nutrition5k
from src.models import SimpleCNN
from src.training import train_model


def _read_dish_ids(path: Path) -> list[str]:
	return [
		dish_id.strip()
		for dish_id in path.read_text(encoding="utf-8-sig").splitlines()
		if dish_id.strip()
	]


def main() -> None:
	parser = argparse.ArgumentParser(description="Train the simple Nutrition5k CNN.")
	parser.add_argument("--epochs", type=int, default=50)
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
	train_dataset = Nutrition5kDataset(
		str(metadata_path),
		str(imagery_root),
		transform=transform,
		dish_ids=_read_dish_ids(Path(TRAIN_IDS_PATH)),
	)
	validation_dataset = Nutrition5kDataset(
		str(metadata_path),
		str(imagery_root),
		transform=transform,
		dish_ids=_read_dish_ids(Path(TEST_IDS_PATH)),
	)
	if not train_dataset or not validation_dataset:
		raise ValueError(
			"Train and validation datasets must both contain images and metadata. "
			"Check the paths and dish ID splits in setup.py."
		)

	loader_options = {
		"batch_size": args.batch_size,
		"num_workers": args.num_workers,
		"collate_fn": collate_nutrition5k,
		"pin_memory": torch.cuda.is_available(),
	}
	train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
	validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

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
	checkpoint_dir = Path(MODEL_CHECKPOINT_DIR) / "simple_cnn"
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	history = train_model(
		model=model,
		train_loader=train_loader,
		optimizer=optimizer,
		criterion=nn.MSELoss(),
		epochs=args.epochs,
		validation_loader=validation_loader,
		last_checkpoint_path=checkpoint_dir / "simple_cnn_latest.pt",
		best_checkpoint_path=checkpoint_dir / "simple_cnn_best.pt",
		model_config=model_config,
	)

	for epoch, (train_loss, validation_loss) in enumerate(
		zip(history["train_loss"], history["val_loss"]), start=1
	):
		print(
			f"Epoch {epoch}/{args.epochs}: "
			f"train_loss={train_loss:.4f}, validation_loss={validation_loss:.4f}"
		)


if __name__ == "__main__":
	main()
