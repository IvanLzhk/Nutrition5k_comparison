import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from setup import IMAGERY_ROOT, METADATA_PATH, TEST_IDS_PATH
from src.dataset.Nutrition5kDataset import Nutrition5kDataset
from src.models import SimpleCNN
from src.training import test_model


TARGET_NAMES = ["calories", "mass_g", "fat_g", "carbs_g", "protein_g"]


def _load_legacy_checkpoint(path: Path) -> Mapping[str, Any]:
	if path.with_suffix(".json").exists():
		raise ValueError(f"Checkpoint already has a JSON sidecar: {path.with_suffix('.json')}")
	checkpoint = torch.load(path, map_location="cpu", weights_only=True)
	if not isinstance(checkpoint, Mapping) or checkpoint.get("format_version") != 1:
		raise ValueError("Unsupported legacy checkpoint format.")
	if checkpoint.get("model_class") != "src.models.simple_cnn.SimpleCNN":
		raise ValueError("This converter only supports SimpleCNN checkpoints.")
	if not isinstance(checkpoint.get("model_config"), Mapping):
		raise ValueError("Legacy checkpoint has no model_config.")
	return checkpoint


def convert_checkpoint(source_path: Path, output_path: Path) -> dict[str, Any]:
	if source_path.resolve() == output_path.resolve():
		raise ValueError("Output path must differ from the source checkpoint.")
	checkpoint = _load_legacy_checkpoint(source_path)
	model_config = dict(checkpoint["model_config"])
	required_config = {"output_features", "image_size", "num_layers", "width"}
	missing_config = required_config - model_config.keys()
	if missing_config:
		raise ValueError(
			"Legacy model_config is missing: " + ", ".join(sorted(missing_config))
		)

	model = SimpleCNN(
		output_features=model_config["output_features"],
		num_layers=model_config["num_layers"],
		width=model_config["width"],
	)
	model.load_state_dict(checkpoint["model_state_dict"])

	optimizer = None
	optimizer_state = checkpoint.get("optimizer_state_dict")
	if optimizer_state is not None:
		adam_name = f"{torch.optim.Adam.__module__}.{torch.optim.Adam.__qualname__}"
		if checkpoint.get("optimizer_class") != adam_name:
			raise ValueError(
				"This converter can preserve optimizer state only for Adam checkpoints."
			)
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(optimizer_state)

	model.save_checkpoint(
		output_path,
		optimizer=optimizer,
		epoch=checkpoint.get("epoch", 0),
		history=checkpoint.get("history", {}),
		best_val_loss=checkpoint.get("best_val_loss"),
		best_epoch=checkpoint.get("best_epoch"),
		model_config=model_config,
		metadata={
			"converted_from": str(source_path.resolve()),
			"legacy_metadata": checkpoint.get("metadata"),
		},
	)
	return model_config


def evaluate_converted_checkpoint(
	checkpoint_path: Path,
	model_config: Mapping[str, Any],
	*,
	batch_size: int,
	num_workers: int,
	accuracy_tolerance_percent: float,
	device: str,
) -> dict[str, Any]:
	model = SimpleCNN(
		output_features=model_config["output_features"],
		num_layers=model_config["num_layers"],
		width=model_config["width"],
	)
	model.load_checkpoint(
		checkpoint_path,
		map_location=device,
		expected_model_config=model_config,
	)
	transform = T.Compose(
		[T.Resize((model_config["image_size"], model_config["image_size"])), T.ToTensor()]
	)
	test_ids = [
		line.strip()
		for line in Path(TEST_IDS_PATH).read_text(encoding="utf-8-sig").splitlines()
		if line.strip()
	]
	test_dataset = Nutrition5kDataset(
		METADATA_PATH,
		IMAGERY_ROOT,
		transform=transform,
		dish_ids=test_ids,
		image_level=True,
	)
	if not test_dataset:
		raise ValueError("The configured test IDs produced an empty test dataset.")
	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=device.startswith("cuda"),
		shuffle=False,
	)
	metrics = test_model(
		model,
		test_loader,
		nn.MSELoss(),
		device=device,
		progress_label="Converted checkpoint test",
		target_names=TARGET_NAMES,
		accuracy_tolerance_percent=accuracy_tolerance_percent,
	)
	metrics_payload = {
		"checkpoint": str(checkpoint_path.resolve()),
		"model_config": dict(model_config),
		"test_dish_count": len(test_ids),
		"accuracy_tolerance_percent": accuracy_tolerance_percent,
		**metrics,
	}
	metrics_path = checkpoint_path.with_name(
		f"{checkpoint_path.stem}_test_metrics.json"
	)
	with metrics_path.open("w", encoding="utf-8") as metrics_file:
		json.dump(metrics_payload, metrics_file, indent=2, sort_keys=True)
		metrics_file.write("\n")
	print(f"Saved test metrics to {metrics_path}")
	print(f"Test MSE: {metrics['loss']:.4f}")
	print(f"Test MAE: {metrics['mae']}")
	print(f"Test MAPE (%): {metrics['mape_percent']}")
	print(f"Test R2: {metrics['r2']}")
	print(f"Test Acc@{accuracy_tolerance_percent:g}% (%): {metrics['acc_at_k_percent']['values_percent']}")
	return metrics


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Convert a legacy SimpleCNN checkpoint and evaluate it on the test split."
	)
	parser.add_argument("--checkpoint", required=True, type=Path)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Converted checkpoint path (default: <checkpoint>_converted.pt).",
	)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--num-workers", type=int, default=8)
	parser.add_argument("--accuracy-tolerance-percent", type=float, default=10.0)
	parser.add_argument(
		"--device",
		default="cuda" if torch.cuda.is_available() else "cpu",
	)
	args = parser.parse_args()
	if not args.checkpoint.is_file():
		parser.error(f"Checkpoint not found: {args.checkpoint}")
	if args.batch_size <= 0 or args.num_workers < 0:
		parser.error("batch-size must be positive and num-workers cannot be negative")
	output_path = args.output or args.checkpoint.with_name(
		f"{args.checkpoint.stem}_converted.pt"
	)
	model_config = convert_checkpoint(args.checkpoint, output_path)
	print(f"Converted checkpoint saved to {output_path}")
	print(f"Metadata saved to {output_path.with_suffix('.json')}")
	evaluate_converted_checkpoint(
		output_path,
		model_config,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		accuracy_tolerance_percent=args.accuracy_tolerance_percent,
		device=args.device,
	)


if __name__ == "__main__":
	main()