import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class Nutrition5kDataset(Dataset):
    def __init__(self, metadata_path: str, imagery_root: str, transform=None, dish_ids=None, image_level=False): # TODO: side angles перевернуті догори дригом, перевернути їх на якомусь етапі
        """
        Args:
            metadata_path: CSV metadata file path (dish_metadata_cafe1.csv).
            imagery_root: Path to the 'imagery' folder (containing 'overhead' and 'side_angles').
            transform: torchvision transforms for images.
            dish_ids: Optional list/set of dish_ids for train/val split.
            image_level: Return each overhead or side image as its own labeled sample.
        """
        self.imagery_root = imagery_root
        self.overhead_dir = os.path.join(imagery_root, "realsense_overhead")
        self.side_angles_dir = os.path.join(imagery_root, "side_angles")
        self.transform = transform
        self.image_level = image_level

        self.entries = []
        dish_filter = set(dish_ids) if dish_ids is not None else None

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue

                dish_id = parts[0]
                if dish_filter and dish_id not in dish_filter:
                    continue

                try:
                    targets = [
                        float(parts[1]),  # calories
                        float(parts[2]),  # mass (g)
                        float(parts[3]),  # fat (g)
                        float(parts[4]),  # carb (g)
                        float(parts[5]),  # protein (g)
                    ]
                except ValueError:
                    continue

                dish_overhead = os.path.join(self.overhead_dir, dish_id)
                dish_side = os.path.join(self.side_angles_dir, dish_id)

                if os.path.isdir(dish_overhead) or os.path.isdir(dish_side):
                    self.entries.append({
                        "dish_id": dish_id,
                        "targets": targets,
                        "overhead_path": dish_overhead,
                        "side_path": dish_side
                    })

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
        self.sample_weights = [
            1.0 / sample_counts[entry["dish_id"]]
            for entry, _, _ in self.image_samples
        ]

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

    def _load_image(self, path: str):
        img = Image.open(path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return T.ToTensor()(img)

    def __getitem__(self, idx):
        if self.image_level:
            entry, image_path, view_type = self.image_samples[idx]
            return {
                "dish_id": entry["dish_id"],
                "view_type": view_type,
                "overhead": self._load_image(image_path),
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
        side_imgs = [self._load_image(p) for p in side_files]
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