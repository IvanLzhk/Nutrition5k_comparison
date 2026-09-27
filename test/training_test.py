import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

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
    def test_base_model_checkpoint_restores_weights_optimizer_and_progress(self):
        model = LinearRegressionModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        predictions = model({"features": torch.ones(1, 1)}).sum()
        predictions.backward()
        optimizer.step()
        expected = model({"features": torch.ones(2, 1)}).detach()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            model.save_checkpoint(
                checkpoint_path,
                optimizer=optimizer,
                epoch=4,
                history={"train_loss": [0.5], "val_loss": [0.25]},
                best_val_loss=0.25,
                best_epoch=4,
                model_config={"input_features": 1},
            )

            restored_model = LinearRegressionModel()
            restored_optimizer = torch.optim.SGD(
                restored_model.parameters(), lr=0.01, momentum=0.9
            )
            progress = restored_model.load_checkpoint(
                checkpoint_path,
                optimizer=restored_optimizer,
                expected_model_config={"input_features": 1},
            )

        actual = restored_model({"features": torch.ones(2, 1)}).detach()
        torch.testing.assert_close(actual, expected)
        self.assertTrue(restored_optimizer.state)
        self.assertEqual(progress["epoch"], 4)
        self.assertEqual(progress["history"], {"train_loss": [0.5], "val_loss": [0.25]})
        self.assertEqual(progress["best_epoch"], 4)

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

    def test_saves_best_and_latest_checkpoints_separately(self):
        model = LinearRegressionModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        validation_losses = iter([0.5, 0.25, 0.75])
        training_epoch = 0

        def fake_run_epoch(model, data_loader, criterion, device, target_key, optimizer=None):
            nonlocal training_epoch
            if optimizer is not None:
                training_epoch += 1
                with torch.no_grad():
                    model.regressor.weight.fill_(training_epoch)
                return 1.0
            return next(validation_losses)

        with tempfile.TemporaryDirectory() as temp_dir:
            last_path = Path(temp_dir) / "last.pt"
            best_path = Path(temp_dir) / "best.pt"
            with patch("src.training.trainer._run_epoch", side_effect=fake_run_epoch):
                train_model(
                    model=model,
                    train_loader=[],
                    optimizer=optimizer,
                    criterion=nn.MSELoss(),
                    epochs=3,
                    validation_loader=[],
                    device="cpu",
                    last_checkpoint_path=last_path,
                    best_checkpoint_path=best_path,
                )

            last_model = LinearRegressionModel()
            last_progress = last_model.load_checkpoint(last_path)
            best_model = LinearRegressionModel()
            best_progress = best_model.load_checkpoint(best_path)

        self.assertEqual(last_progress["epoch"], 3)
        self.assertEqual(best_progress["epoch"], 2)
        self.assertEqual(best_progress["best_val_loss"], 0.25)
        torch.testing.assert_close(
            last_model.regressor.weight,
            torch.full_like(last_model.regressor.weight, 3.0),
        )
        torch.testing.assert_close(
            best_model.regressor.weight,
            torch.full_like(best_model.regressor.weight, 2.0),
        )

    def test_resume_runs_additional_epochs_and_appends_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "resume.pt"
            model = LinearRegressionModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            model.save_checkpoint(
                checkpoint_path,
                optimizer=optimizer,
                epoch=7,
                history={"train_loss": [1.0], "val_loss": [0.5]},
                best_val_loss=0.5,
                best_epoch=7,
            )

            resumed_model = LinearRegressionModel()
            resumed_optimizer = torch.optim.SGD(
                resumed_model.parameters(), lr=0.01, momentum=0.9
            )
            loader = DataLoader(RegressionDataset(), batch_size=4)
            history = train_model(
                model=resumed_model,
                train_loader=loader,
                optimizer=resumed_optimizer,
                criterion=nn.MSELoss(),
                epochs=1,
                validation_loader=loader,
                device="cpu",
                resume_from=checkpoint_path,
                last_checkpoint_path=checkpoint_path,
            )
            progress = resumed_model.load_checkpoint(checkpoint_path)

        self.assertEqual(len(history["train_loss"]), 2)
        self.assertEqual(len(history["val_loss"]), 2)
        self.assertEqual(progress["epoch"], 8)

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
