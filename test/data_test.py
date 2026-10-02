import csv
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.experiments.cnn_experiment import CNNExperiment
from src.dataset.Nutrition5kDataset import Nutrition5kDataset, collate_nutrition5k
from main import build_train_augmentation



class Nutrition5kDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.metadata_path = self.root / "metadata.csv"
        self.imagery_root = self.root / "imagery"

        rows = [
            ["dish-1", "100", "200", "10", "20", "30"],
            ["dish-2", "150", "250", "15", "25", "35"],
        ]
        with self.metadata_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerows(rows)

        self._create_dish("dish-1", side_count=1)
        self._create_dish("dish-2", side_count=2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_dish(self, dish_id, side_count):
        overhead_dir = self.imagery_root / "realsense_overhead" / dish_id
        side_dir = self.imagery_root / "side_angles" / dish_id / "frames_sampled25"
        overhead_dir.mkdir(parents=True)
        side_dir.mkdir(parents=True)

        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(overhead_dir / "rgb.png")
        for index in range(side_count):
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(
                side_dir / f"frame_{index}.png"
            )

    def _make_dataset(
        self, cache_dir=None, image_level=False, transform=None, augmentation=None
    ):
        return Nutrition5kDataset(
            metadata_path=str(self.metadata_path),
            imagery_root=str(self.imagery_root),
            transform=transform or T.Compose([T.Resize((4, 4)), T.ToTensor()]),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            image_level=image_level,
            augmentation=augmentation,
        )

    def test_image_level_dataset_returns_each_view_as_a_sample(self):
        dataset = Nutrition5kDataset(
            metadata_path=str(self.metadata_path),
            imagery_root=str(self.imagery_root),
            transform=T.Compose([T.Resize((4, 4)), T.ToTensor()]),
            image_level=True,
        )

        self.assertEqual(len(dataset), 5)
        self.assertEqual(
            [dataset[index]["view_type"] for index in range(len(dataset))],
            ["overhead", "side", "overhead", "side", "side"],
        )
        dish_weights = {}
        for index, (entry, _, _) in enumerate(dataset.image_samples):
            dish_weights[entry["dish_id"]] = (
                dish_weights.get(entry["dish_id"], 0.0)
                + dataset.sample_weights[index]
            )
        self.assertEqual(dish_weights, {"dish-1": 1.0, "dish-2": 1.0})

    def test_loads_entries_and_applies_image_transform(self):
        dataset = self._make_dataset()

        self.assertEqual(len(dataset), 2)
        sample = dataset[0]

        self.assertEqual(sample["dish_id"], "dish-1")
        self.assertEqual(sample["overhead"].shape, (3, 4, 4))
        self.assertEqual(sample["side_views"].shape, (1, 3, 4, 4))
        torch.testing.assert_close(
            sample["targets"], torch.tensor([100, 200, 10, 20, 30], dtype=torch.float32)
        )

    def test_image_level_flips_side_images_but_not_overhead(self):
        side_path = (
            self.imagery_root
            / "side_angles"
            / "dish-1"
            / "frames_sampled25"
            / "frame_0.png"
        )
        image = Image.new("RGB", (8, 8))
        for y in range(8):
            color = (255, 0, 0) if y < 4 else (0, 0, 255)
            for x in range(8):
                image.putpixel((x, y), color)
        image.save(side_path)

        dataset = self._make_dataset(image_level=True)
        overhead = dataset[0]["overhead"]
        side = dataset[1]["overhead"]
        with Image.open(side_path) as source:
            expected_side = T.Compose(
                [T.Resize((4, 4)), T.ToTensor()]
            )(source.convert("RGB"))

        torch.testing.assert_close(side, torch.flip(expected_side, dims=[1]))
        self.assertFalse(torch.equal(overhead, side))

    def test_image_cache_reuses_and_invalidates_transformed_side_images(self):
        cache_dir = self.root / "cache"
        dataset = self._make_dataset(cache_dir=cache_dir, image_level=True)

        first = dataset[1]["overhead"]
        cache_files = list(cache_dir.glob("*.png"))
        self.assertEqual(len(cache_files), 1)
        cached_mtime = cache_files[0].stat().st_mtime_ns

        cached = dataset[1]["overhead"]
        torch.testing.assert_close(cached, first)
        self.assertEqual(cache_files[0].stat().st_mtime_ns, cached_mtime)

        source_path = Path(dataset.image_samples[1][1])
        source_stat = source_path.stat()
        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(source_path)
        os.utime(
            source_path,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000_000),
        )

        updated = dataset[1]["overhead"]
        self.assertFalse(torch.equal(updated, first))
        self.assertEqual(len(list(cache_dir.glob("*.png"))), 2)

    def test_image_cache_key_includes_transform_settings(self):
        cache_dir = self.root / "cache"
        self._make_dataset(cache_dir=cache_dir, image_level=True)[1]
        small_transform = T.Compose([T.Resize((2, 2)), T.ToTensor()])
        small_dataset = self._make_dataset(
            cache_dir=cache_dir, image_level=True, transform=small_transform
        )

        small_image = small_dataset[1]["overhead"]

        self.assertEqual(small_image.shape, (3, 2, 2))
        self.assertEqual(len(list(cache_dir.glob("*.png"))), 2)

    def test_image_cache_skips_tensors_not_representable_as_png(self):
        cache_dir = self.root / "cache"
        normalized_transform = T.Compose(
            [
                T.Resize((4, 4)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        dataset = self._make_dataset(
            cache_dir=cache_dir,
            image_level=True,
            transform=normalized_transform,
        )

        image = dataset[0]["overhead"]

        self.assertEqual(image.shape, (3, 4, 4))
        self.assertTrue(torch.isfinite(image).all())
        self.assertEqual(list(cache_dir.glob("*.png")), [])

    def test_augmentation_runs_after_cache_load_on_every_access(self):
        cache_dir = self.root / "cache"
        call_count = 0

        def add_call_count(tensor):
            nonlocal call_count
            call_count += 1
            return tensor + call_count

        dataset = self._make_dataset(
            cache_dir=cache_dir,
            image_level=True,
            augmentation=add_call_count,
        )

        first = dataset[1]["overhead"]
        second = dataset[1]["overhead"]

        torch.testing.assert_close(second - first, torch.ones_like(first))

    def test_cnn_augmentation_changes_train_images_without_expanding_batches(self):
        side_path = (
            self.imagery_root
            / "side_angles"
            / "dish-1"
            / "frames_sampled25"
            / "frame_0.png"
        )
        image = Image.new("RGB", (8, 8))
        for y in range(8):
            for x in range(8):
                image.putpixel((x, y), (255, 0, 0) if x < 4 else (0, 0, 255))
        image.save(side_path)

        experiment = object.__new__(CNNExperiment)
        experiment.metadata_path = self.metadata_path
        experiment.imagery_root = self.imagery_root
        experiment.transform = T.Compose([T.Resize((4, 4)), T.ToTensor()])
        experiment.cache_dir = None
        experiment.train_augmentation = T.RandomHorizontalFlip(p=1.0)
        train_dataset = experiment._create_dataset(["dish-1"], training=True)
        evaluation_dataset = experiment._create_dataset(["dish-1"])

        self.assertEqual(len(train_dataset), len(evaluation_dataset))
        self.assertIsNotNone(train_dataset.augmentation)
        self.assertIsNone(evaluation_dataset.augmentation)
        self.assertFalse(
            torch.equal(
                train_dataset[0]["side_views"][0],
                evaluation_dataset[0]["side_views"][0],
            )
        )

        experiment.batch_size = 2
        experiment.num_workers = 0
        experiment.collate_fn = collate_nutrition5k
        batch = next(iter(experiment._create_loader(train_dataset)))
        self.assertEqual(batch["overhead"].shape, (1, 3, 4, 4))
        self.assertEqual(batch["side_views"].shape, (1, 3, 4, 4))

    def test_experiment_loader_persists_workers_when_enabled(self):
        experiment = object.__new__(CNNExperiment)
        experiment.batch_size = 2
        experiment.num_workers = 1

        loader = experiment._create_loader(self._make_dataset())
        self.assertTrue(loader.persistent_workers)

        experiment.num_workers = 0
        loader = experiment._create_loader(self._make_dataset())
        self.assertFalse(loader.persistent_workers)

    def test_main_builds_default_train_augmentation(self):
        augmentation = build_train_augmentation()

        self.assertIsInstance(augmentation, T.Compose)
        self.assertEqual(len(augmentation.transforms), 3)

    def test_collate_handles_different_numbers_of_side_views(self):
        loader = DataLoader(
            self._make_dataset(),
            batch_size=2,
            collate_fn=collate_nutrition5k,
        )

        batch = next(iter(loader))

        self.assertEqual(batch["batch_size"], 2)
        self.assertEqual(batch["overhead"].shape, (2, 3, 4, 4))
        self.assertEqual(batch["side_views"].shape, (3, 3, 4, 4))
        torch.testing.assert_close(
            batch["side_dish_indices"], torch.tensor([0, 1, 1], dtype=torch.long)
        )
        self.assertFalse(torch.isnan(batch["targets"]).any())

    def test_collate_uses_side_image_when_overhead_is_missing(self):
        (self.imagery_root / "realsense_overhead" / "dish-1" / "rgb.png").unlink()
        loader = DataLoader(
            self._make_dataset(),
            batch_size=2,
            collate_fn=collate_nutrition5k,
        )

        batch = next(iter(loader))

        torch.testing.assert_close(batch["overhead"][0], batch["side_views"][0])
        self.assertEqual(batch["overhead"].shape, (2, 3, 4, 4))

    def test_collate_handles_missing_side_images(self):
        side_dir = self.imagery_root / "side_angles" / "dish-1" / "frames_sampled25"
        for image_path in side_dir.iterdir():
            image_path.unlink()
        loader = DataLoader(
            self._make_dataset(),
            batch_size=2,
            collate_fn=collate_nutrition5k,
        )

        batch = next(iter(loader))

        self.assertEqual(batch["side_views"].shape, (2, 3, 4, 4))
        torch.testing.assert_close(
            batch["side_dish_indices"], torch.tensor([1, 1], dtype=torch.long)
        )

    def test_collate_handles_batch_without_any_side_images(self):
        for dish_id in ("dish-1", "dish-2"):
            side_dir = self.imagery_root / "side_angles" / dish_id / "frames_sampled25"
            for image_path in side_dir.iterdir():
                image_path.unlink()
        loader = DataLoader(
            self._make_dataset(),
            batch_size=2,
            collate_fn=collate_nutrition5k,
        )

        batch = next(iter(loader))

        self.assertEqual(batch["side_views"].shape, (0, 3, 4, 4))
        self.assertEqual(batch["side_dish_indices"].shape, (0,))

    def test_side_image_selection_limits_each_camera_and_total(self):
        side_dir = self.imagery_root / "side_angles" / "dish-1" / "frames_sampled25"
        for image_path in side_dir.iterdir():
            image_path.unlink()
        for camera in "ABCD":
            for index in range(15):
                (side_dir / f"camera_{camera}_frame_{index:03d}.png").touch()

        dataset = self._make_dataset()
        selected = dataset._get_image_paths(str(side_dir))

        self.assertEqual(len(selected), 40)
        for camera in "ABCD":
            self.assertLessEqual(
                sum(
                    f"camera_{camera}_" in Path(path).name
                    for path in selected
                ),
                10,
            )
        self.assertEqual(selected, dataset._get_image_paths(str(side_dir)))

    def test_side_image_selection_falls_back_to_total_limit(self):
        side_dir = self.imagery_root / "side_angles" / "dish-1" / "frames_sampled25"
        for index in range(50):
            (side_dir / f"frame_{index:03d}.png").touch()

        dataset = self._make_dataset()

        self.assertEqual(len(dataset._get_image_paths(str(side_dir))), 40)

    def test_metadata_rejects_duplicate_dish_ids(self):
        duplicate_path = self.root / "duplicate_metadata.csv"
        duplicate_path.write_text(
            "dish-1,100,200,10,20,30\n"
            "dish-1,101,201,11,21,31\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Duplicate dish ID"):
            Nutrition5kDataset(
                metadata_path=str(duplicate_path),
                imagery_root=str(self.imagery_root),
                transform=T.ToTensor(),
            )


if __name__ == "__main__":
    unittest.main()
