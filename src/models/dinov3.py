from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models import vit_b_16

from .base_model import BaseModel


class DinoV3(BaseModel):
    """DINOv3 image encoder with a nutrition regression head.

    Pass a local Hugging Face model directory containing ``config.json`` and
    ``model.safetensors`` through ``weights`` when ``pretrained`` is true.
    The directory is loaded locally through ``transformers``.
    """

    def __init__(
        self,
        output_features: int = 5,
        *,
        model_name: str = "dinov3_vitb16",
        weights: str | Path | None = None,
        pretrained: bool = True,
        encode_batch_size: int = 16,
        repository: str | Path = "D:/vs_projects/dinov3",
        backbone: nn.Module | None = None,
        feature_channels: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(output_features, bool) or not isinstance(output_features, int):
            raise ValueError("output_features must be a positive integer.")
        if output_features < 1:
            raise ValueError("output_features must be a positive integer.")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")
        if isinstance(encode_batch_size, bool) or not isinstance(encode_batch_size, int):
            raise ValueError("encode_batch_size must be a positive integer.")
        if encode_batch_size < 1:
            raise ValueError("encode_batch_size must be a positive integer.")
        if backbone is None:
            if pretrained:
                if weights is None or (
                    isinstance(weights, (str, Path)) and not str(weights).strip()
                ):
                    raise ValueError(
                        "A local DINOv3 Hugging Face model directory is required "
                        "when pretrained is true."
                    )
                weights_path = Path(weights).expanduser().resolve()
                if not weights_path.is_dir():
                    raise FileNotFoundError(
                        f"DINOv3 Hugging Face model directory was not found: "
                        f"{weights_path}"
                    )
                if not (weights_path / "config.json").is_file() or not (
                    weights_path / "model.safetensors"
                ).is_file():
                    raise FileNotFoundError(
                        "DINOv3 weights must be a Hugging Face model directory "
                        "containing config.json and model.safetensors."
                    )
                try:
                    from transformers import AutoModel
                except ImportError as error:
                    raise ImportError(
                        "Loading DINOv3 safetensors requires the transformers "
                        "package."
                    ) from error
                backbone = AutoModel.from_pretrained(
                    str(weights_path),
                    local_files_only=True,
                )
            else:
                # This path keeps unit tests and shape experiments offline. It is
                # not a DINOv3 checkpoint and must not be used for comparisons.
                backbone = vit_b_16(weights=None)
                backbone.heads = nn.Identity()

        inferred_channels = getattr(backbone, "embed_dim", None)
        if inferred_channels is None:
            inferred_channels = getattr(backbone, "num_features", None)
        if inferred_channels is None:
            inferred_channels = getattr(backbone, "hidden_dim", None)
        if inferred_channels is None:
            config = getattr(backbone, "config", None)
            inferred_channels = getattr(config, "hidden_size", None)
        if inferred_channels is None and feature_channels is None:
            raise ValueError(
                "feature_channels is required for a DINOv3 backbone without "
                "embed_dim or num_features."
            )
        self.backbone = backbone
        self.feature_channels = feature_channels or int(inferred_channels)
        self.encode_batch_size = encode_batch_size
        self.model_name = model_name
        self.repository = str(repository)
        self.pretrained = pretrained
        self.regressor = nn.Linear(self.feature_channels, output_features)
        self._backbone_trainable = True

    def set_backbone_trainable(self, trainable: bool) -> None:
        self._backbone_trainable = trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
        if not trainable:
            self.backbone.eval()

    def train(self, mode: bool = True) -> "DinoV3":
        super().train(mode)
        if not self._backbone_trainable:
            self.backbone.eval()
        return self

    def forward(self, batch: Mapping[str, Any]) -> Tensor:
        if "image" in batch:
            images = batch["image"]
        elif "overhead" in batch:
            images = batch["overhead"]
        else:
            raise KeyError("Батч повинен містити ключ 'image' або 'overhead'")

        features = self._encode_images(images)
        return self.regressor(features)
    
    def _encode_images(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError("Images must have shape (batch, 3, height, width).")
        if images.size(0) <= self.encode_batch_size:
            return self._encode_batch(images)
        return torch.cat(
            [
                self._encode_batch(images[start : start + self.encode_batch_size])
                for start in range(0, images.size(0), self.encode_batch_size)
            ],
            dim=0,
        )

    def _encode_batch(self, images: Tensor) -> Tensor:
        if hasattr(self.backbone, "forward_features"):
            output = self.backbone.forward_features(images)
            if isinstance(output, Mapping):
                for key in ("x_norm_clstoken", "cls_token", "x_prenorm"):
                    if key in output:
                        output = output[key]
                        break
            if isinstance(output, (tuple, list)):
                output = output[0]
        else:
            if getattr(getattr(self.backbone, "config", None), "model_type", None) == "dinov3_vit":
                output = self.backbone(pixel_values=images)
            else:
                output = self.backbone(images)
            if hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state
        if not isinstance(output, Tensor):
            raise TypeError("DINOv3 backbone must return a tensor feature embedding.")
        if output.ndim == 3:
            output = output[:, 0]
        if output.ndim != 2 or output.size(1) != self.feature_channels:
            raise ValueError(
                "DINOv3 backbone must return features with shape "
                f"(batch, {self.feature_channels})."
            )
        return output


DinoV3Model = DinoV3
