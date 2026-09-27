import argparse
import os
from pathlib import Path
import tempfile
from typing import Iterable


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "nutrition5k_dataset"


def _dish_id_from_line(line: str) -> str:
    return line.split(",", 1)[0].strip()


def _has_overhead_image(dish_folder: Path) -> bool:
    return (dish_folder / "rgb.png").is_file()


def _has_side_image(dish_folder: Path) -> bool:
    frames_folder = dish_folder / "frames_sampled25"
    target_folder = frames_folder if frames_folder.is_dir() else dish_folder
    if not target_folder.is_dir():
        return False
    return any(
        image.is_file()
        and not image.name.startswith(".")
        and image.suffix.lower() in IMAGE_EXTENSIONS
        for image in target_folder.iterdir()
    )


def _read_lines(path: Path) -> tuple[list[str], bool]:
    content = path.read_bytes()
    has_bom = content.startswith(b"\xef\xbb\xbf")
    text = content.decode("utf-8-sig")
    return text.splitlines(keepends=True), has_bom


def _write_lines(path: Path, lines: Iterable[str], has_bom: bool) -> None:
    content = "".join(lines).encode("utf-8")
    if has_bom:
        content = b"\xef\xbb\xbf" + content

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f"{path.name}.", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _filter_file(path: Path, removed_ids: set[str], *, apply: bool) -> int:
    lines, has_bom = _read_lines(path)
    retained_lines = [line for line in lines if _dish_id_from_line(line) not in removed_ids]
    removed_count = len(lines) - len(retained_lines)
    if apply and removed_count:
        _write_lines(path, retained_lines, has_bom)
    return removed_count


def _find_empty_directories(root: Path) -> list[Path]:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    removable: set[Path] = set()
    for directory in directories:
        children = list(directory.iterdir())
        if all(child.is_dir() and child in removable for child in children):
            removable.add(directory)
    return sorted(removable, key=lambda path: len(path.parts), reverse=True)


def clean_dataset(dataset_root: Path, *, apply: bool) -> tuple[set[str], list[Path]]:
    imagery_root = dataset_root / "imagery"
    metadata_root = dataset_root / "metadata"
    dish_ids_root = dataset_root / "dish_ids"
    for required_path in (imagery_root, metadata_root, dish_ids_root):
        if not required_path.is_dir():
            raise FileNotFoundError(f"Required dataset directory not found: {required_path}")

    metadata_files = sorted(metadata_root.glob("dish_metadata_*.csv"))
    id_files = sorted(dish_ids_root.rglob("*.txt"))
    all_dish_ids = {
        _dish_id_from_line(line)
        for path in metadata_files + id_files
        for line in _read_lines(path)[0]
        if _dish_id_from_line(line)
    }

    overhead_root = imagery_root / "realsense_overhead"
    side_root = imagery_root / "side_angles"
    removed_ids = {
        dish_id
        for dish_id in all_dish_ids
        if not _has_overhead_image(overhead_root / dish_id)
        and not _has_side_image(side_root / dish_id)
    }

    for path in metadata_files + id_files:
        count = _filter_file(path, removed_ids, apply=apply)
        if count:
            action = "Removed" if apply else "Would remove"
            print(f"{action} {count} entries from {path}")

    empty_directories = _find_empty_directories(imagery_root)
    if apply:
        for directory in empty_directories:
            directory.rmdir()
        action = "Removed"
    else:
        action = "Would remove"
    for directory in empty_directories:
        print(f"{action} empty directory: {directory}")

    return removed_ids, empty_directories


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove dish IDs with no overhead or side images and prune empty "
            "directories from the Nutrition5k dataset."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Nutrition5k dataset root (defaults to this script's parent dataset directory).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the cleanup. Without this flag, only print the planned changes.",
    )
    args = parser.parse_args()

    removed_ids, empty_directories = clean_dataset(args.dataset_root, apply=args.apply)
    mode = "Applied" if args.apply else "Dry run"
    print(
        f"{mode}: {len(removed_ids)} dish IDs have no usable images; "
        f"{len(empty_directories)} empty directories found."
    )
    if not args.apply:
        print("Re-run with --apply to modify metadata, dish ID lists, and directories.")


if __name__ == "__main__":
    main()