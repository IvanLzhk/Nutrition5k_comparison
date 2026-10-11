"""Run one-image inference and report forward-pass resource usage."""

import ctypes
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from PIL import Image
from torch.profiler import ProfilerActivity, profile
import torchvision.transforms as T

try:
    import resource
except ImportError:  # pragma: no cover - only used on Windows
    resource = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    ResNet18,
    SimpleCNN,
    SwinTransformerSmall,
    SwinTransformerTiny,
)


TARGET_NAMES = ("calories", "mass_g", "fat_g", "carbs_g", "protein_g")
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
MODEL_CLASSES = {
    "src.models.simple_cnn.SimpleCNN": SimpleCNN,
    "src.models.resnet18.ResNet18": ResNet18,
    "src.models.swin_transformer_tiny.SwinTransformerTiny": SwinTransformerTiny,
    "src.models.swin_transformer_small.SwinTransformerSmall": SwinTransformerSmall,
}
CHECKPOINT_PATH = Path("C:/Users/Admin/projects/Nutrition5k_comparison/checkpoints/swin_transformer_tiny/2026.10.05_21-10-09/single_split/swin_transformer_tiny_best.pt")
IMAGE_PATH = Path("C:/Users/Admin/projects/Nutrition5k_comparison/example.jpg")
DEVICE = "auto"


def _load_checkpoint_metadata(checkpoint_path: Path) -> Mapping[str, Any]:
    metadata_path = checkpoint_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"Checkpoint metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, Mapping) or metadata.get("format_version") != 1:
        raise ValueError("Unsupported or invalid checkpoint metadata.")
    model_class = metadata.get("model_class")
    if model_class not in MODEL_CLASSES:
        raise ValueError(f"Unsupported checkpoint model class: {model_class!r}")
    model_config = metadata.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("Checkpoint metadata does not contain model_config.")
    required = {"output_features", "image_size", "target_mean", "target_std"}
    missing = required - model_config.keys()
    if missing:
        raise ValueError(
            "Checkpoint model_config is missing: " + ", ".join(sorted(missing))
        )
    if model_config["output_features"] != len(TARGET_NAMES):
        raise ValueError("This script expects five nutritional output features.")
    return metadata


def _build_model(metadata: Mapping[str, Any]) -> torch.nn.Module:
    model_class = metadata["model_class"]
    config = metadata["model_config"]
    model_type = MODEL_CLASSES[model_class]
    if model_type is SimpleCNN:
        return SimpleCNN(
            output_features=config["output_features"],
            num_layers=config["num_layers"],
            width=config["width"],
        )
    if model_type is ResNet18:
        return ResNet18(output_features=config["output_features"], weights=None)
    if model_type is SwinTransformerSmall:
        return SwinTransformerSmall(
            output_features=config["output_features"],
            weights=None,
        )
    return SwinTransformerTiny(
        output_features=config["output_features"],
        weights=None,
    )


def _process_memory_bytes() -> int:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
        success = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if not success:
            error_code = ctypes.get_last_error()
            raise OSError(error_code, "GetProcessMemoryInfo failed.")
        return counters.working_set_size
    if resource is None:
        raise OSError("Process memory reporting is unavailable on this platform.")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss * (1024 if os.name != "darwin" else 1)


def _run_profiled_forward(
    model: torch.nn.Module, inputs: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, int]:
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
    ) as profiler:
        with torch.inference_mode():
            predictions = model(inputs)
    flops = sum(event.flops for event in profiler.key_averages())
    if flops <= 0:
        raise RuntimeError("The profiler did not produce a FLOPs estimate.")
    return predictions, int(flops)


def inverse_standardize(
    predictions: torch.Tensor,
    target_mean: list[float],
    target_std: list[float],
) -> torch.Tensor:
    mean = torch.tensor(target_mean, dtype=predictions.dtype, device=predictions.device)
    std = torch.tensor(target_std, dtype=predictions.dtype, device=predictions.device)
    return predictions * std + mean


def run_inference(
    checkpoint_path: Path,
    image_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    metadata = _load_checkpoint_metadata(checkpoint_path)
    model = _build_model(metadata).to(device)
    model.load_checkpoint(checkpoint_path, map_location=device)
    model.eval()

    config = metadata["model_config"]
    transform = T.Compose(
        [
            T.Resize((config["image_size"], config["image_size"])),
            T.ToTensor(),
            T.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    with Image.open(image_path) as image:
        input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    inputs = {"image": input_tensor}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    process_memory_before = _process_memory_bytes() if device.type == "cpu" else 0
    predictions, flops = _run_profiled_forward(model, inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memory = torch.cuda.max_memory_allocated(device)
    else:
        memory = max(process_memory_before, _process_memory_bytes())
    predictions = inverse_standardize(
        predictions[0],
        config["target_mean"],
        config["target_std"],
    ).detach().cpu()
    return predictions, flops, memory


def _parse_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    try:
        device = _parse_device(DEVICE)
        predictions, flops, memory = run_inference(
            CHECKPOINT_PATH, IMAGE_PATH, device
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(str(error)) from error

    print(f"Device: {device}")
    print("Predictions:")
    for name, value in zip(TARGET_NAMES, predictions.tolist()):
        print(f"  {name}: {value:.4f}")
    print(f"Forward FLOPs: {flops:,}")
    memory_label = "peak CUDA allocated memory" if device.type == "cuda" else "process RSS"
    print(f"{memory_label}: {memory / (1024 ** 2):.2f} MiB")


if __name__ == "__main__":
    main()
