import unittest
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models import BaseModel
from src.training import get_kfold_splits, train_model


class RegressionDataset(Dataset):
    def __init__(self):
        self.features = torch.arange(1, 9, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index]
        return {"features": features, "targets": features * 2}


class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.regressor = nn.Linear(1, 1)

    def forward(self, batch):
        if "targets" in batch:
            raise AssertionError("Targets must not be passed to the model.")
        return self.regressor(batch["features"])


class TrainModelTests(unittest.TestCase):
    def test_trains_model_and_reports_validation_loss(self):
        loader = DataLoader(RegressionDataset(), batch_size=4)
        model = LinearRegressionModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]

        history = train_model(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            criterion=nn.MSELoss(),
            epochs=3,
            validation_loader=loader,
            device="cpu",
        )

        self.assertEqual(len(history["train_loss"]), 3)
        self.assertEqual(len(history["val_loss"]), 3)
        self.assertTrue(all(torch.isfinite(torch.tensor(loss)) for loss in history["train_loss"]))
        self.assertEqual(next(model.parameters()).device.type, "cpu")
        self.assertTrue(
            any(
                not torch.equal(initial, current)
                for initial, current in zip(initial_parameters, model.parameters())
            )
        )

    def test_kfold_splits_partition_training_ids_once(self):
        dish_ids = [f"dish-{index}" for index in range(11)]
        with tempfile.TemporaryDirectory() as temp_dir:
            train_ids_path = Path(temp_dir) / "train_ids.txt"
            train_ids_path.write_text("\n".join(dish_ids), encoding="utf-8")

            split_iterator = get_kfold_splits(
                n_splits=5,
                shuffle=True,
                random_state=21,
                train_ids_path=train_ids_path,
            )
            splits = list(split_iterator)

        self.assertEqual(len(splits), 5) # expected folds
        self.assertEqual([len(val_ids) for _, val_ids in splits], [3, 2, 2, 2, 2]) # expected split
        self.assertCountEqual(
            (dish_id for _, val_ids in splits for dish_id in val_ids),
            dish_ids,
        )# all ids present in val sets
        for train_ids, val_ids in splits:
            self.assertFalse(set(train_ids) & set(val_ids)) # no id in both train and val
            self.assertEqual(set(train_ids) | set(val_ids), set(dish_ids)) # all ids present

    def test_rejects_mismatched_prediction_shape(self):
        class WrongShapeModel(BaseModel):
            def forward(self, batch):
                return batch["features"].squeeze(1)

        loader = DataLoader(RegressionDataset(), batch_size=4)
        model = WrongShapeModel()
        optimizer = torch.optim.SGD([nn.Parameter(torch.tensor(1.0))], lr=0.01)

        with self.assertRaisesRegex(ValueError, "matching shapes"):
            train_model(model, loader, optimizer, nn.MSELoss(), epochs=1, device="cpu")


if __name__ == "__main__":
    unittest.main()
