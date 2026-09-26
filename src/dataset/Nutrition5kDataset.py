import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class Nutrition5kDataset(Dataset):
    def __init__(self, metadata_path: str, imagery_root: str, transform=None):
        self.overhead_dir = os.path.join(imagery_root, "overhead")
        self.side_angles_dir = os.path.join(imagery_root, "side_angles")
        self.transform = transform
        
        self.entries = []
        with open(metadata_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                dish_id = parts[0]
                try:
                    targets = [float(p) for p in parts[1:6]] # [cal, mass, fat, carb, protein]
                except ValueError:
                    continue

                dish_overhead = os.path.join(self.overhead_dir, dish_id)
                dish_side = os.path.join(self.side_angles_dir, dish_id)

                if os.path.isdir(dish_overhead) and os.path.isdir(dish_side):
                    self.entries.append({
                        "dish_id": dish_id,
                        "targets": targets,
                        "overhead_path": dish_overhead,
                        "side_path": dish_side
                    })

    def __len__(self):
        return len(self.entries)

    def _get_images(self, folder: str):
        valid_exts = (".png", ".jpg", ".jpeg")
        files = [
            os.path.join(folder, f) 
            for f in sorted(os.listdir(folder)) 
            if f.lower().endswith(valid_exts) and not f.startswith(".")
        ]
        return files

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return T.ToTensor()(img)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # 1. overhead
        overhead_files = self._get_images(entry["overhead_path"])
        if not overhead_files:
            raise FileNotFoundError(f"No overhead image: {entry['dish_id']}")
        overhead_tensor = self._load_image(overhead_files[0])

        side_files = self._get_images(entry["side_path"])
        if not side_files:
            side_files = [overhead_files[0]]
            
        side_tensors = [self._load_image(p) for p in side_files]
        side_tensor = torch.stack(side_tensors, dim=0)

        return {
            "dish_id": entry["dish_id"],
            "overhead": overhead_tensor,                  # (C, H, W)
            "side_views": side_tensor,                    # (V_i, C, H, W)
            "targets": torch.tensor(entry["targets"], dtype=torch.float32) # (5,)
        }