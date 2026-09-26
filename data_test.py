import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image, ImageDraw

from src.dataset.Nutrition5kDataset import Nutrition5kDataset, collate_nutrition5k


def visualize_batch(batch, max_side_views=6):
    overhead = batch["overhead"].detach().cpu()
    side_views = batch["side_views"].detach().cpu()
    side_dish_indices = batch["side_dish_indices"].detach().cpu()
    batch_size = batch["batch_size"]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def to_display_image(tensor):
        image = (tensor * std + mean).clamp(0, 1)
        return T.ToPILImage()(image)

    image_height, image_width = overhead.shape[-2:]
    label_height = 24
    gap = 8
    columns = max_side_views + 1
    canvas = Image.new(
        "RGB",
        (gap + columns * (image_width + gap), gap + batch_size * (image_height + label_height + gap)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for dish_idx in range(batch_size):
        row_y = gap + dish_idx * (image_height + label_height + gap)
        dish_id = batch["dish_ids"][dish_idx]
        x = gap
        draw.text((x, row_y), f"{dish_id} - overhead", fill="black")
        canvas.paste(to_display_image(overhead[dish_idx]), (x, row_y + label_height))

        matching_indices = torch.where(side_dish_indices == dish_idx)[0][:max_side_views]
        for view_number, view_idx in enumerate(matching_indices, start=1):
            x = gap + view_number * (image_width + gap)
            draw.text((x, row_y), f"side {view_number}", fill="black")
            canvas.paste(to_display_image(side_views[view_idx]), (x, row_y + label_height))

    canvas.show(title="Nutrition5k batch tensors")


def verify_pipeline():
    METADATA_PATH = "src\\dataset\\nutrition5k_dataset\\metadata\\dish_metadata_cafe1.csv"
    IMAGERY_ROOT = "src\\dataset\\nutrition5k_dataset\\imagery"

    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("--> Initializing dataset...")
    dataset = Nutrition5kDataset(
        metadata_path=METADATA_PATH,
        imagery_root=IMAGERY_ROOT,
        transform=transforms
    )
    print(f"Dishes: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: Empty dataset. Please check the metadata and imagery paths.")
        return

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_nutrition5k
    )

    print("--> first batch...")
    batch = next(iter(loader))

    dish_ids = batch["dish_ids"]
    overhead = batch["overhead"]
    side_views = batch["side_views"]
    indices = batch["side_dish_indices"]
    targets = batch["targets"]
    B = batch["batch_size"]

    print("\n" + "=" * 45)
    print("RESULTS OF BATCH VERIFICATION:")
    print("=" * 45)
    print(f"Batch size (B):            {B}")
    print(f"Dish IDs in batch:         {dish_ids}")
    print(f"Overhead tensor:           {overhead.shape} -> expected [B, 3, 224, 224]")
    print(f"Side Views tensor:         {side_views.shape} -> [Total_V, 3, 224, 224]")
    print(f"Side Dish Indices:         {indices.shape} -> [Total_V]")
    print(f"Targets tensor:            {targets.shape} -> [B, 5]")
    print("=" * 45)

    print("\nside-views:")
    counts = torch.bincount(indices, minlength=B)
    for i in range(B):
        print(f" - Страва #{i} ({dish_ids[i]}): {counts[i].item()} бічних фото")

    assert overhead.shape[0] == B, f"Overhead batch size error: {overhead.shape[0]} != {B}"
    assert targets.shape == (B, 5), f"Targets shape error: {targets.shape}"
    assert side_views.shape[0] == indices.shape[0], "side_views shape error!"
    assert indices.max().item() < B and indices.min().item() >= 0, "side_dish_indices out of bounds!"

    # 3. Check NaN / Inf
    assert not torch.isnan(overhead).any(), "Overhead NaN!"
    assert not torch.isnan(side_views).any(), "Side views NaN!"
    assert not torch.isnan(targets).any(), "Targets NaN!"

    # 4. Приклад тарґетів
    print("\n[cal, mass(g), fat(g), carb(g), protein(g)]:")
    for i in range(B):
        vals = [f"{v:.1f}" for v in targets[i].tolist()]
        print(f" - {dish_ids[i]}: {vals}")

    visualize_batch(batch)
    print("\n[OK]")
    input("Press Enter to continue...")

if __name__ == "__main__":
    verify_pipeline()