import argparse

from setup import BATCH_SIZE, NUM_WORKERS
from src.experiments.cnn_experiment import CNNExperiment


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
	CNNExperiment(parser.parse_args()).run()


if __name__ == "__main__":
	main()
