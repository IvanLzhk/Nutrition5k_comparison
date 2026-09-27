import random
from pathlib import Path
from typing import Iterator, Optional, Union

from setup import TRAIN_IDS_PATH


def get_kfold_splits(
    n_splits: int = 5,
    shuffle: bool = False,
    random_state: Optional[int] = None,
    train_ids_path: Optional[Union[str, Path]] = None,
) -> Iterator[tuple[list[str], list[str]]]:
    """Return ``(train_ids, val_ids)`` partitions of the configured training IDs."""
    if isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    ids_path = Path(train_ids_path) if train_ids_path is not None else Path(TRAIN_IDS_PATH)
    ids = [line.strip() for line in ids_path.read_text(encoding="utf-8-sig").splitlines()]
    ids = [dish_id for dish_id in ids if dish_id]
    if n_splits > len(ids):
        raise ValueError("n_splits cannot exceed the number of training IDs.")

    indices = list(range(len(ids)))
    if shuffle:
        random.Random(random_state).shuffle(indices)

    fold_sizes = [len(ids) // n_splits] * n_splits
    for index in range(len(ids) % n_splits):
        fold_sizes[index] += 1

    start = 0
    for fold_size in fold_sizes:
        val_indices = set(indices[start : start + fold_size])
        train_ids = [dish_id for index, dish_id in enumerate(ids) if index not in val_indices]
        val_ids = [ids[index] for index in indices[start : start + fold_size]]
        yield train_ids, val_ids
        start += fold_size
