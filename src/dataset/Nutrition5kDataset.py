import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class Nutrition5kDataset(Dataset):
    def __init__(self, metadata_path: str, imagery_root: str, transform=None, dish_ids=None):
        """
        Args:
            metadata_path: шлях до CSV файлу метаданих (dish_metadata_cafe1.csv).
            imagery_root: шлях до папки 'imagery' (що містить 'overhead' та 'side_angles').
            transform: torchvision трансформації для зображень.
            dish_ids: необов'язковий список/множина dish_id для train/val спліту.
        """
        self.imagery_root = imagery_root
        self.overhead_dir = os.path.join(imagery_root, "realsense_overhead")
        self.side_angles_dir = os.path.join(imagery_root, "side_angles")
        self.transform = transform

        self.entries = []
        dish_filter = set(dish_ids) if dish_ids is not None else None

        # Зчитування метаданих
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
                    # Пропуск заголовка або битих рядків
                    continue

                dish_overhead = os.path.join(self.overhead_dir, dish_id)
                dish_side = os.path.join(self.side_angles_dir, dish_id)

                # Перевірка наявності обох директорій
                if os.path.isdir(dish_overhead) and os.path.isdir(dish_side):
                    self.entries.append({
                        "dish_id": dish_id,
                        "targets": targets,
                        "overhead_path": dish_overhead,
                        "side_path": dish_side
                    })

    def __len__(self):
        return len(self.entries)

    def _get_image_paths(self, dish_folder: str):
        valid_exts = (".jpeg", ".jpg", ".png")
        
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
        entry = self.entries[idx]

        # 1. Завантаження RGB overhead кадру
        overhead_path = os.path.join(entry["overhead_path"], "rgb.png")
        if not os.path.isfile(overhead_path):
            raise FileNotFoundError(f"Немає overhead RGB фото для: {entry['dish_id']}")
        overhead_img = self._load_image(overhead_path)  # (C, H, W)

        # 2. Завантаження всіх наявних side_angles кадрів (V_i штук)
        side_files = self._get_image_paths(entry["side_path"])
        if not side_files:
            # Fallback: якщо раптом фото ракурсів відсутні, дублюємо overhead
            side_files = [overhead_path]

        side_imgs = [self._load_image(p) for p in side_files]
        side_tensor = torch.stack(side_imgs, dim=0)  # (V_i, C, H, W)

        targets_tensor = torch.tensor(entry["targets"], dtype=torch.float32)  # (5,)

        return {
            "dish_id": entry["dish_id"],
            "overhead": overhead_img,
            "side_views": side_tensor,
            "targets": targets_tensor
        }


def collate_nutrition5k(batch):
    """
    Кастомний collate:
    - overhead: звичайний батч (B, C, H, W)
    - side_views: конкатенація всіх ракурсів батчу (Total_V, C, H, W)
    - side_dish_indices: вектор довжини Total_V з ID страви в батчі (0 .. B-1)
    - targets: (B, 5)
    """
    batch_size = len(batch)
    dish_ids = [item["dish_id"] for item in batch]

    overhead = torch.stack([item["overhead"] for item in batch], dim=0)
    targets = torch.stack([item["targets"] for item in batch], dim=0)

    side_views_list = []
    dish_indices = []

    for b_idx, item in enumerate(batch):
        v = item["side_views"]  # (V_i, C, H, W)
        side_views_list.append(v)
        # Записуємо індекс елемента батчу для кожного ракурсу цієї страви
        dish_indices.extend([b_idx] * v.size(0))

    side_views = torch.cat(side_views_list, dim=0)
    side_dish_indices = torch.tensor(dish_indices, dtype=torch.long)

    return {
        "dish_ids": dish_ids,
        "overhead": overhead,
        "side_views": side_views,
        "side_dish_indices": side_dish_indices,
        "targets": targets,
        "batch_size": batch_size
    }