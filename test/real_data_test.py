import unittest
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from setup import METADATA_PATH, IMAGERY_ROOT
from src.dataset.Nutrition5kDataset import (
    Nutrition5kDataset,
    collate_nutrition5k,
)


class RealDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata_path = Path(METADATA_PATH)
        cls.imagery_root = Path(IMAGERY_ROOT)

        if not cls.metadata_path.is_file():
            raise unittest.SkipTest(
                f"Metadata file not found: {cls.metadata_path}"
            )

        if not cls.imagery_root.is_dir():
            raise unittest.SkipTest(
                f"Imagery directory not found: {cls.imagery_root}"
            )

    def test_loads_real_batch(self):
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        dataset = Nutrition5kDataset(
            metadata_path=str(self.metadata_path),
            imagery_root=str(self.imagery_root),
            transform=transforms,
        )

        self.assertGreater(
            len(dataset),
            0,
            "The real dataset contains no usable entries.",
        )

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_nutrition5k,
        )

        batch = next(iter(loader))

        batch_size = batch["batch_size"]
        overhead = batch["overhead"]
        side_views = batch["side_views"]
        side_indices = batch["side_dish_indices"]
        targets = batch["targets"]

        self.assertEqual(overhead.shape, (batch_size, 3, 224, 224))
        self.assertEqual(targets.shape, (batch_size, 5))
        self.assertEqual(side_views.shape[0], side_indices.shape[0])

        self.assertTrue(
            torch.all((side_indices >= 0) & (side_indices < batch_size))
        )

        self.assertTrue(torch.isfinite(overhead).all())
        self.assertTrue(torch.isfinite(side_views).all())
        self.assertTrue(torch.isfinite(targets).all())


if __name__ == "__main__":
    unittest.main()