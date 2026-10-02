import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class Nutrition5kDataset(Dataset):
    CACHE_VERSION = 3

    def __init__(
        self,
        metadata_path: str,
        imagery_root: str,
        transform=None,
        dish_ids=None,
        image_level=False,
        cache_dir=None,
        augmentation=None,
        post_transform=None,
    ):
        """
        Args:
            metadata_path: CSV metadata file path (dish_metadata_cafe1.csv).
            imagery_root: Path to the 'imagery' folder (containing 'overhead' and 'side_angles').
            transform: torchvision transforms for images.
            dish_ids: Optional list/set of dish_ids for train/val split.
            image_level: Return each overhead or side image as its own labeled sample.
        """
        self.imagery_root = imagery_root
        self.imagery_root_path = Path(imagery_root).resolve()
        self.overhead_dir = os.path.join(imagery_root, "realsense_overhead")
        self.side_angles_dir = os.path.join(imagery_root, "side_angles")
        self.transform = transform
        self.image_level = image_level
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.augmentation = augmentation
        self.post_transform = post_transform

        self.entries = []
        dish_filter = set(dish_ids) if dish_ids is not None else None

        seen_dish_ids = set()
        with open(metadata_path, "r", newline="", encoding="utf-8-sig") as f:
            for line_number, parts in enumerate(csv.reader(f), start=1):
                if not parts or not any(part.strip() for part in parts):
                    continue
                if parts[0].strip().lower() in {"dish_id", "dish id"}:
                    continue
                if len(parts) < 6:
                    raise ValueError(
                        f"Metadata row {line_number} has fewer than six columns."
                    )

                dish_id = parts[0].strip()
                if not dish_id:
                    raise ValueError(f"Metadata row {line_number} has an empty dish ID.")
                if dish_id in seen_dish_ids:
                    raise ValueError(f"Duplicate dish ID in metadata: {dish_id!r}.")
                seen_dish_ids.add(dish_id)
                if dish_filter and dish_id not in dish_filter:
                    continue

                try:
                    targets = [float(value) for value in parts[1:6]]
                except ValueError as error:
                    raise ValueError(
                        f"Metadata row {line_number} has non-numeric targets."
                    ) from error
                if not all(math.isfinite(value) and value >= 0 for value in targets):
                    raise ValueError(
                        f"Metadata row {line_number} has invalid target values."
                    )

                dish_overhead = os.path.join(self.overhead_dir, dish_id)
                dish_side = os.path.join(self.side_angles_dir, dish_id)

                if os.path.isdir(dish_overhead) or os.path.isdir(dish_side):
                    self.entries.append({
                        "dish_id": dish_id,
                        "targets": targets,
                        "overhead_path": dish_overhead,
                        "side_path": dish_side
                    })

        unusable_entries = [
            entry["dish_id"]
            for entry in self.entries
            if not os.path.isfile(os.path.join(entry["overhead_path"], "rgb.png"))
            and not self._get_image_paths(entry["side_path"])
        ]
        if unusable_entries:
            raise ValueError(
                "Dishes without usable imagery: "
                f"{unusable_entries[:5]}"
            )

        self.image_samples = []
        sample_counts = {}
        if self.image_level:
            for entry in self.entries:
                dish_samples = []
                overhead_path = os.path.join(entry["overhead_path"], "rgb.png")
                if os.path.isfile(overhead_path):
                    dish_samples.append((entry, overhead_path, "overhead"))
                dish_samples.extend(
                    (entry, path, "side")
                    for path in self._get_image_paths(entry["side_path"])
                )
                self.image_samples.extend(dish_samples)
                sample_counts[entry["dish_id"]] = len(dish_samples)
        if self.image_level:
            self.sample_weights = [
                1.0 / sample_counts[entry["dish_id"]]
                for entry, _, _ in self.image_samples
            ]
        else:
            self.sample_weights = [1.0] * len(self.entries)

    def __len__(self):
        return len(self.image_samples) if self.image_level else len(self.entries)

    def _get_image_paths(self, dish_folder: str):
        valid_exts = (".jpeg", ".jpg", ".png")

        if not os.path.isdir(dish_folder):
            return []

        frames_dir = os.path.join(dish_folder, "frames_sampled25")
        target_dir = frames_dir if os.path.isdir(frames_dir) else dish_folder

        paths = [
            os.path.join(target_dir, f)
            for f in sorted(os.listdir(target_dir))
            if f.lower().endswith(valid_exts) and not f.startswith(".")
        ]
        return paths

    def _cache_path(self, path: str, flip_vertical: bool) -> Path | None:
        if self.cache_dir is None:
            return None

        source_path = Path(path).resolve()
        source_stat = source_path.stat()
        cache_identity = {
            "version": self.CACHE_VERSION,
            "imagery_root": str(self.imagery_root_path),
            "source": os.path.relpath(source_path, self.imagery_root_path),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "transform": repr(self.transform),
            "flip_vertical": flip_vertical,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{cache_key}.png"

    def _load_image(self, path: str, flip_vertical: bool = False):
        cache_path = self._cache_path(path, flip_vertical)
        if cache_path is not None and cache_path.is_file():
            try:
                with Image.open(cache_path) as cached_image:
                    tensor = T.ToTensor()(cached_image.convert("RGB"))
            except OSError:
                cache_path.unlink(missing_ok=True)
            else:
                return self._apply_image_transforms(tensor)

        with Image.open(path) as image_file:
            img = image_file.convert("RGB")
        tensor = self.transform(img) if self.transform else T.ToTensor()(img)
        if flip_vertical:
            tensor = torch.flip(tensor, dims=[1])

        can_cache = (
            cache_path is not None
            and isinstance(tensor, torch.Tensor)
            and tensor.ndim == 3
            and tensor.size(0) == 3
            and torch.isfinite(tensor).all()
            and tensor.min().item() >= 0
            and tensor.max().item() <= 1
        )
        if can_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=cache_path.parent,
                    prefix=f"{cache_path.stem}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                T.ToPILImage()(tensor.detach().cpu()).save(
                    temporary_path, format="PNG"
                )
                os.replace(temporary_path, cache_path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        return self._apply_image_transforms(tensor)

    def _apply_image_transforms(self, tensor):
        if self.augmentation:
            tensor = self.augmentation(tensor)
        if self.post_transform:
            tensor = self.post_transform(tensor)
        return tensor

    def __getitem__(self, idx):
        if self.image_level:
            entry, image_path, view_type = self.image_samples[idx]
            return {
                "dish_id": entry["dish_id"],
                "view_type": view_type,
                "overhead": self._load_image(
                    image_path, flip_vertical=view_type == "side"
                ),
                "targets": torch.tensor(entry["targets"], dtype=torch.float32),
            }

        entry = self.entries[idx]

        # RGB overhead
        overhead_path = os.path.join(entry["overhead_path"], "rgb.png")
        if not os.path.isfile(overhead_path):
            overhead_img = None
        else:
            overhead_img = self._load_image(overhead_path)  # (C, H, W)

        # side_angles
        side_files = self._get_image_paths(entry["side_path"])
        side_imgs = [self._load_image(p, flip_vertical=True) for p in side_files]
        side_tensor = (
            torch.stack(side_imgs, dim=0)
            if side_imgs
            else torch.empty((0,))
        )

        targets_tensor = torch.tensor(entry["targets"], dtype=torch.float32)

        return {
            "dish_id": entry["dish_id"],
            "overhead": overhead_img,
            "side_views": side_tensor,
            "targets": targets_tensor
        }


def collate_nutrition5k(batch):
    """
    Custom collate for DataLoader, because different number of side_views per dish.
    """
    batch_size = len(batch)
    dish_ids = [item["dish_id"] for item in batch]

    overhead_images = []
    side_views_list = []
    dish_indices = []

    for batch_index, item in enumerate(batch):
        overhead_image = item["overhead"]
        side_views = item["side_views"]
        if overhead_image is None:
            if side_views.ndim != 4 or side_views.size(0) == 0:
                raise ValueError(
                    f"Dish {item['dish_id']!r} has neither an overhead nor a side image."
                )
            overhead_image = side_views[0]
        overhead_images.append(overhead_image)

        if side_views.ndim == 4 and side_views.size(0) > 0:
            side_views_list.append(side_views)
            dish_indices.extend([batch_index] * side_views.size(0))

    overhead = torch.stack(overhead_images, dim=0)
    targets = torch.stack([item["targets"] for item in batch], dim=0)
    side_views = (
        torch.cat(side_views_list, dim=0)
        if side_views_list
        else overhead.new_empty((0, *overhead.shape[1:]))
    )
    side_dish_indices = torch.tensor(dish_indices, dtype=torch.long)

    return {
        "dish_ids": dish_ids,
        "overhead": overhead,
        "side_views": side_views,
        "side_dish_indices": side_dish_indices,
        "targets": targets,
        "batch_size": batch_size
    }