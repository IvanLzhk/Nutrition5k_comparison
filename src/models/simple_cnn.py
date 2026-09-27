from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from .base_model import BaseModel


class SimpleCNN(BaseModel):
    """Small overhead-image regressor for exercising the training pipeline."""

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
        self.regressor = nn.Linear(input_channels, output_features)

    def forward(self, batch: Mapping[str, Any]) -> Tensor:
        return self.regressor(self.features(batch["overhead"]).flatten(start_dim=1))