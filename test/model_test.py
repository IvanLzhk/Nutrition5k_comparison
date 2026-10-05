import unittest

import torch

from src.models import ResNet18, SimpleCNN, SwinTransformerTiny


class SwinTransformerTinyTests(unittest.TestCase):
    def test_frozen_backbone_stays_in_eval_mode(self):
        model = SwinTransformerTiny(output_features=5, weights=None)
        model.set_backbone_trainable(False)
        model.train()

        self.assertFalse(model.backbone.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.backbone.parameters())
        )

    def test_unfrozen_backbone_returns_to_train_mode(self):
        model = SwinTransformerTiny(output_features=5, weights=None)
        model.set_backbone_trainable(False)
        model.set_backbone_trainable(True)
        model.train()

        self.assertTrue(model.backbone.training)
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.backbone.parameters())
        )


class SimpleCNNMultiViewTests(unittest.TestCase):
    def test_forward_accepts_one_image(self):
        model = SimpleCNN(output_features=5, num_layers=1, width=2)
        predictions = model({"image": torch.rand(2, 3, 8, 8)})

        self.assertEqual(predictions.shape, (2, 5))
        self.assertTrue(torch.isfinite(predictions).all())

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
	def test_forward_accepts_one_image(self):
		model = ResNet18(output_features=5)
		predictions = model({"image": torch.rand(2, 3, 64, 64)})

		self.assertEqual(predictions.shape, (2, 5))
		self.assertTrue(torch.isfinite(predictions).all())

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
