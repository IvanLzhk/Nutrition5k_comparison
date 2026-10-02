import unittest

import torch

from src.models import ResNet18, SimpleCNN


class SimpleCNNMultiViewTests(unittest.TestCase):
    def test_forward_pools_side_views_per_dish(self):
        model = SimpleCNN(output_features=5, num_layers=1, width=2)
        batch = {
            "overhead": torch.rand(2, 3, 8, 8),
            "side_views": torch.rand(3, 3, 8, 8),
            "side_dish_indices": torch.tensor([0, 1, 1]),
        }

        predictions = model(batch)

        self.assertEqual(predictions.shape, (2, 5))
        self.assertTrue(torch.isfinite(predictions).all())

    def test_forward_accepts_dishes_without_side_views(self):
        model = SimpleCNN(output_features=5, num_layers=1, width=2)
        batch = {
            "overhead": torch.rand(2, 3, 8, 8),
            "side_views": torch.empty(0, 3, 8, 8),
            "side_dish_indices": torch.empty(0, dtype=torch.long),
        }

        predictions = model(batch)

        self.assertEqual(predictions.shape, (2, 5))
        self.assertTrue(torch.isfinite(predictions).all())


class ResNet18MultiViewTests(unittest.TestCase):
	def test_forward_pools_side_views_per_dish(self):
		model = ResNet18(output_features=5)
		batch = {
			"overhead": torch.rand(2, 3, 64, 64),
			"side_views": torch.rand(3, 3, 64, 64),
			"side_dish_indices": torch.tensor([0, 1, 1]),
		}

		predictions = model(batch)

		self.assertEqual(predictions.shape, (2, 5))
		self.assertTrue(torch.isfinite(predictions).all())

	def test_forward_accepts_dishes_without_side_views(self):
		model = ResNet18(output_features=5)
		batch = {
			"overhead": torch.rand(2, 3, 64, 64),
			"side_views": torch.empty(0, 3, 64, 64),
			"side_dish_indices": torch.empty(0, dtype=torch.long),
		}

		predictions = model(batch)

		self.assertEqual(predictions.shape, (2, 5))
		self.assertTrue(torch.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
