import csv
import unittest
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from setup import METADATA_PATH, IMAGERY_ROOT, TRAIN_IDS_PATH, TEST_IDS_PATH
from src.dataset.Nutrition5kDataset import (
    Nutrition5kDataset,
)


class RealDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata_path = Path(METADATA_PATH)
        cls.imagery_root = Path(IMAGERY_ROOT)
        cls.train_ids_path = Path(TRAIN_IDS_PATH)
        cls.test_ids_path = Path(TEST_IDS_PATH)

        if not cls.metadata_path.is_file():
            raise unittest.SkipTest(
                f"Metadata file not found: {cls.metadata_path}"
            )

        if not cls.imagery_root.is_dir():
            raise unittest.SkipTest(
                f"Imagery directory not found: {cls.imagery_root}"
            )

        for split_path in (cls.train_ids_path, cls.test_ids_path):
            if not split_path.is_file():
                raise unittest.SkipTest(f"Dish ID split file not found: {split_path}")

    def test_rgb_split_ids_exist_in_configured_metadata(self): #make sure all ids in train/test splits exist in metadata
        with self.metadata_path.open(newline="", encoding="utf-8-sig") as metadata_file:
            metadata_ids = {row[0] for row in csv.reader(metadata_file) if row}

        for split_name, split_path in (
            ("train", self.train_ids_path),
            ("test", self.test_ids_path),
        ):
            split_ids = {
                line.strip()
                for line in split_path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            } # read lines and make dict
            missing_ids = split_ids - metadata_ids
            self.assertFalse(
                missing_ids,
                f"{len(missing_ids)} {split_name} IDs are missing from "
                f"{self.metadata_path.name}: {sorted(missing_ids)[:10]}",
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
            image_level=True,
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
        )

        batch = next(iter(loader))

        images = batch["image"]
        targets = batch["targets"]

        self.assertEqual(images.shape, (2, 3, 224, 224))
        self.assertEqual(targets.shape, (2, 5))
        self.assertTrue(torch.isfinite(images).all())
        self.assertTrue(torch.isfinite(targets).all())


if __name__ == "__main__":
    unittest.main()