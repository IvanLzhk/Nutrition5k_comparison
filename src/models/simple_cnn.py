from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .base_model import BaseModel


class SimpleCNN(BaseModel):
    """Small image regressor that predicts from one view at a time."""

    def __init__(
        self,
        output_features: int = 5,
        num_layers: int = 3,
        width: int = 16,
    ) -> None:
        super().__init__()
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be a positive integer.")
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer.")

        feature_layers = []
        input_channels = 3
        for layer_index in range(num_layers):
            output_channels = width * (2 ** layer_index)
            feature_layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                ]
            )
            if layer_index < num_layers - 1:
                feature_layers.append(nn.MaxPool2d(2))
            input_channels = output_channels
        feature_layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*feature_layers)
        self.feature_channels = input_channels
        self.regressor = nn.Linear(input_channels * 2, output_features)

    def forward(self, batch: Mapping[str, Any]) -> Tensor:
        if "image" in batch:
            image = batch["image"]
            if image.ndim != 4 or image.size(1) != 3:
                raise ValueError("image must have shape (batch, 3, height, width).")
            features = self._encode_images(image)
            return self.regressor(torch.cat([features, features], dim=1))

        # Keep loading older multi-view checkpoints and callers working.
        overhead_features = self._encode_images(batch["overhead"])
        side_views = batch["side_views"]
        side_dish_indices = batch["side_dish_indices"]
        if side_views.ndim != 4 or side_dish_indices.ndim != 1:
            raise ValueError(
                "side_views must be a 4D tensor and side_dish_indices must be 1D."
            )
        if side_views.size(0) != side_dish_indices.numel():
            raise ValueError(
                "side_views and side_dish_indices must contain the same number "
                "of images."
            )

        side_features = overhead_features.new_zeros(overhead_features.shape)
        if side_views.size(0) > 0:
            encoded_side_views = self._encode_images(side_views)
            if side_dish_indices.numel() and (
                side_dish_indices.min() < 0
                or side_dish_indices.max() >= overhead_features.size(0)
            ):
                raise ValueError("side_dish_indices contains an invalid dish index.")
            side_features.index_add_(0, side_dish_indices, encoded_side_views)
            side_counts = overhead_features.new_zeros(
                (overhead_features.size(0), 1)
            )
            side_counts.index_add_(
                0,
                side_dish_indices,
                overhead_features.new_ones((side_dish_indices.numel(), 1)),
            )
            side_features = side_features / side_counts.clamp_min(1)

        return self.regressor(torch.cat([overhead_features, side_features], dim=1))

    def _encode_images(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError("Images must have shape (batch, 3, height, width).")
        return self.features(images).flatten(start_dim=1)