from pathlib import Path
import tempfile
import unittest

import torch
from PIL import Image

from scripts.run_model import inverse_standardize, run_inference
from src.models import SimpleCNN


class InferenceTests(unittest.TestCase):
    def test_inverse_standardize_restores_original_units(self):
        predictions = inverse_standardize(
            torch.zeros(1, 5), [10, 20, 30, 40, 50], [2, 3, 4, 5, 6]
        )
        self.assertTrue(
            torch.equal(predictions, torch.tensor([[10, 20, 30, 40, 50]]))
        )

    def test_runs_checkpoint_inference_and_reports_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            model = SimpleCNN(num_layers=1, width=2)
            config = {
                "output_features": 5,
                "image_size": 8,
                "num_layers": 1,
                "width": 2,
                "target_mean": [1, 2, 3, 4, 5],
                "target_std": [1, 1, 1, 1, 1],
            }
            model.save_checkpoint(checkpoint, model_config=config)
            image_path = root / "image.png"
            Image.new("RGB", (12, 12), color=(128, 64, 32)).save(image_path)

            predictions, flops, memory = run_inference(
                checkpoint, image_path, torch.device("cpu")
            )

            self.assertEqual(predictions.shape, (5,))
            self.assertGreater(flops, 0)
            self.assertGreater(memory, 0)

    def test_missing_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({}, checkpoint)
            with self.assertRaisesRegex(ValueError, "metadata not found"):
                run_inference(checkpoint, checkpoint, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
