from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models import Swin_T_Weights, swin_t

from .base_model import BaseModel


class SwinTransformerTiny(BaseModel):
    """Pretrained Swin Transformer Tiny backbone with a regression head."""

    def __init__(
        self,
        output_features: int = 5,
        *,
        weights: Swin_T_Weights | None = Swin_T_Weights.DEFAULT,
        encode_batch_size: int = 8,
    ) -> None:
        super().__init__()
        if isinstance(output_features, bool) or not isinstance(output_features, int):
            raise ValueError("output_features must be a positive integer.")
        if output_features < 1:
            raise ValueError("output_features must be a positive integer.")
        if isinstance(encode_batch_size, bool) or not isinstance(encode_batch_size, int):
            raise ValueError("encode_batch_size must be a positive integer.")
        if encode_batch_size < 1:
            raise ValueError("encode_batch_size must be a positive integer.")

        backbone = swin_t(weights=weights)
        feature_channels = backbone.head.in_features
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.feature_channels = feature_channels
        self.encode_batch_size = encode_batch_size
        self.regressor = nn.Linear(feature_channels * 2, output_features)
        self._backbone_trainable = True

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Enable or disable gradient updates for the pretrained backbone."""
        self._backbone_trainable = trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
        if not trainable:
            self.backbone.eval()

    def train(self, mode: bool = True) -> "SwinTransformerTiny":
        super().train(mode)
        if not self._backbone_trainable:
            self.backbone.eval()
        return self

    def forward(self, batch: Mapping[str, Any]) -> Tensor:
        if "image" in batch:
            image = batch["image"]
            if image.ndim != 4 or image.size(1) != 3:
                raise ValueError("image must have shape (batch, 3, height, width).")
            features = self._encode_images(image)
            return self.regressor(torch.cat([features, features], dim=1))

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
        if images.size(0) <= self.encode_batch_size:
            return self.backbone(images)
        return torch.cat(
            [
                self.backbone(images[start : start + self.encode_batch_size])
                for start in range(0, images.size(0), self.encode_batch_size)
            ],
            dim=0,
        )


SwinTiny = SwinTransformerTiny
