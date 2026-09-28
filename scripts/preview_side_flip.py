import argparse
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw, ImageOps
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.Nutrition5kDataset import Nutrition5kDataset


SIDE_ROOT = PROJECT_ROOT / "data" / "nutrition5k_dataset" / "imagery" / "side_angles"
IMAGERY_ROOT = PROJECT_ROOT / "data" / "nutrition5k_dataset" / "imagery"
METADATA_PATH = PROJECT_ROOT / "data" / "nutrition5k_dataset" / "metadata" / "dish_metadata_cafe1.csv"
VALID_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def get_images(folder: Path) -> list[Path]:
    frames_folder = folder / "frames_sampled25"
    source_folder = frames_folder if frames_folder.is_dir() else folder
    return sorted(
        path
        for path in source_folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS and not path.name.startswith(".")
    )


def find_source_image(dish_id: str | None, image_path: Path | None, frame_index: int) -> Path:
    if image_path is not None:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image_path

    if dish_id is not None:
        dish_folder = SIDE_ROOT / dish_id
        images = get_images(dish_folder) if dish_folder.is_dir() else []
    else:
        images = []
        for dish_folder in sorted(path for path in SIDE_ROOT.iterdir() if path.is_dir()):
            images = get_images(dish_folder)
            if images:
                break

    if not images:
        raise FileNotFoundError(f"No side images found under {SIDE_ROOT}")
    if frame_index < 0 or frame_index >= len(images):
        raise IndexError(f"Frame index {frame_index} is out of range (found {len(images)} images)")
    return images[frame_index]


def save_comparison(source_path: Path, output_path: Path, image_size: int) -> None:
    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    dataset = Nutrition5kDataset(
        str(METADATA_PATH), str(IMAGERY_ROOT), transform=transform
    )
    original_tensor = dataset._load_image(str(source_path))
    flipped_tensor = torch.flip(original_tensor, dims=[1])
    to_image = T.ToPILImage()
    original = to_image(original_tensor)
    flipped = to_image(flipped_tensor)
    panel_size = (640, 480)
    label_height = 36
    padding = 16
    gap = 16
    canvas = Image.new(
        "RGB",
        (padding * 2 + panel_size[0] * 2 + gap, padding * 2 + label_height + panel_size[1]),
        "#202423",
    )
    draw = ImageDraw.Draw(canvas)

    for index, (label, image) in enumerate((("Original", original), ("Vertically flipped", flipped))):
        left = padding + index * (panel_size[0] + gap)
        draw.text((left, padding + 8), label, fill="white")
        preview = ImageOps.contain(image, panel_size)
        top = padding + label_height + (panel_size[1] - preview.height) // 2
        canvas.paste(preview, (left + (panel_size[0] - preview.width) // 2, top))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Source image: {source_path}")
    print(f"Comparison saved to: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a side image before and after a vertical flip.")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--dish-id", help="Dish folder to preview; defaults to the first dish with an image.")
    source_group.add_argument("--image", type=Path, help="Path to a specific side image.")
    parser.add_argument("--frame-index", type=int, default=0, help="Zero-based frame index within the selected dish.")
    parser.add_argument("--image-size", type=int, default=128, help="Resize to the model input size before flipping.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "side_flip_preview.png",
        help="Where to save the side-by-side preview.",
    )
    args = parser.parse_args()

    source_path = find_source_image(args.dish_id, args.image, args.frame_index)
    save_comparison(source_path, args.output, args.image_size)


if __name__ == "__main__":
    main()
