# Nutrition5k Model Comparison

A personal, work-in-progress project for comparing computer vision models on the [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) dataset.

The main goal is to make model comparisons reproducible while keeping the training pipeline flexible enough to support different standard architectures, training strategies, and experiment configurations with minimal changes.

> **Status:** Active development. The project structure and training workflow are usable, but results, APIs, and configuration details may still change.

## Highlights

- Compare multiple models through a common experiment interface.
- Currently configured examples include:
  - `SimpleCNN`
  - `ResNet18`
  - `Swin Transformer Tiny`
  - `Swin Transformer Small`
  - `DINOv3 ViT-B/16` (local DINOv3 checkpoint)
- Support for overhead images and optional side-angle images for each dish.
- Per-dish aggregation of side-view features before regression.
- Flexible single-split or K-fold training configuration.
- Shared training and evaluation logic across experiments.
- Image transformation caching to reduce repeated preprocessing.
- Configurable data augmentation, batch size, learning rate, image size, and other hyperparameters.
- Training conveniences including:
  - Early stopping
  - Automatic mixed precision (AMP)
  - Gradient clipping
  - Gradient accumulation
  - Checkpoint saving and optional checkpoint resuming
- Evaluation metrics including MSE, MAE, MAPE, R², and accuracy within a configurable tolerance.

## What is being predicted?

The pipeline is designed as a multi-output regression task. For each dish, the model predicts five nutritional targets:

- Calories
- Mass
- Fat
- Carbohydrates
- Protein

Targets are standardized during training and converted back to the original scale for evaluation.

## Project structure

```text
.
├── main.py                         # Configure and run model experiments
├── setup.py                        # Dataset, checkpoint, and loader settings
├── run_tests.py                    # Test discovery and execution
├── docs/
│   └── architecture.md             # Training and evaluation architecture
├── scripts/
│   ├── clean_empty_data.py         # Dataset cleanup utility
│   ├── convert_and_test_checkpoint.py
│   ├── run_inference.py            # Single-photo inference and resource report
│   └── preview_side_flip.py        # Image preprocessing/debugging utility
├── src/
│   ├── dataset/
│   │   └── Nutrition5kDataset.py   # Dataset and collator implementation
│   ├── experiments/                 # Model-specific experiment wrappers
│   ├── models/                      # Model implementations
│   └── training/                    # Trainer and split utilities
└── test/                            # Unit tests
```

For a visual description of the current training flow, see [`docs/architecture.md`](docs/architecture.md).

## Data setup

Download and prepare the Nutrition5k dataset separately. The current configuration expects the following files and directories:

```text
data/
└── nutrition5k_dataset/
    ├── imagery/
    ├── metadata/
    │   └── dish_metadata_cafe1.csv
    └── dish_ids/
        └── splits/
            ├── rgb_train_ids.txt
            └── rgb_test_ids.txt
```

The default paths are defined in [`setup.py`](setup.py). If your dataset is stored elsewhere, update those paths before running an experiment.

The project may create the following directories while running:

```text
data/cache/nutrition5k/   # Cached transformed images
checkpoints/                # Model checkpoints and metadata
```

## Installation

Create a Python environment and install the dependencies required by the project. A CUDA-enabled PyTorch installation is recommended for practical training times.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate       # Windows

python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pandas pillow scikit-learn transformers
```

Depending on the local environment, additional development or visualization dependencies may be useful for running the helper scripts and tests.

## Running experiments

After placing the dataset at the configured paths, run:

```bash
python main.py
```

`main.py` runs the configured experiments sequentially. The default configuration compares the models listed in the **Highlights** section using shared settings where possible.

Before starting a run, review the experiment configuration near the top of `main.py`, including:

- `EPOCHS`
- `FOLDS`
- `BATCH_SIZE`
- `IMAGE_SIZE`
- Model-specific learning rates
- `EARLY_STOPPING_PATIENCE`
- `GRADIENT_ACCUMULATION_STEPS`
- `USE_AMP`
- `RESUME_FROM`

For example, to resume from a previous checkpoint, set `RESUME_FROM` to the checkpoint path:

```python
RESUME_FROM = Path(
    "checkpoints/simple_cnn/<timestamp>/single_split/simple_cnn_best.pt"
)
```

## Adding another model

The pipeline is intended to make adding standard models straightforward:

1. Add the model implementation under `src/models/`.
2. Export it from `src/models/__init__.py` if appropriate.
3. Add or reuse an experiment wrapper under `src/experiments/`.
4. Register the model and experiment factory in `main.py`.
5. Reuse the shared dataset, trainer, checkpoint, and evaluation components.

This separation between models, experiments, data handling, and training is the central design goal of the project. It allows the same evaluation workflow to be applied to different architectures without duplicating the complete training loop.

### DINOv3

DINOv3 is loaded through the local DINOv3 repository configured by
`DINOV3_REPOSITORY`. Install Transformers and set `DINOV3_WEIGHTS` in
`setup.py` to the downloaded Hugging Face snapshot directory:

```python
DINOV3_WEIGHTS = (
    r"C:\Users\admin\.cache\huggingface\hub\models--facebook--"
    r"dinov3-vitb16-pretrain-lvd1689m\snapshots\<snapshot-id>"
)
```

The directory must contain `config.json` and `model.safetensors`. The default
value is `None`; running with it unchanged fails early with a configuration
error.

## Outputs

Training runs save artifacts under `checkpoints/`, including:

- Latest training-state checkpoints
- Best validation checkpoints
- Checkpoint metadata
- Training history
- Final `test_metrics.json` files

The image transformation cache is stored under `data/cache/nutrition5k/`.

## Running single-photo inference

Use `scripts/run_inference.py` with a checkpoint produced by this project and
one photo:

```bash
python scripts/run_inference.py \
  --checkpoint checkpoints/resnet18/<timestamp>/single_split/resnet18_best.pt \
  --image path/to/photo.jpg
```

The script supports `SimpleCNN`, `ResNet18`, and `Swin Transformer Tiny`
checkpoints. It reads the model class, architecture settings, input image size,
and target standardization statistics from the checkpoint's sibling `.json`
metadata file, so the checkpoint and metadata file must both be present.
Predictions are printed in the original units for calories, mass, fat,
carbohydrates, and protein. Use `--device cpu`, `--device cuda`, or leave the
default `--device auto` to select CUDA when available.

The resource report describes one image forward pass. `Forward FLOPs` is the
PyTorch profiler's operator FLOPs estimate. CUDA runs report peak allocated
device memory; CPU runs report the process resident memory observed around the
forward pass. FLOPs profiling uses PyTorch's built-in profiler and does not
require an additional package.

## Running tests

Run the repository test suite with:

```bash
python run_tests.py
```

## Reproducibility notes

The current configuration includes a fixed random seed and records experiment metadata alongside checkpoints. For more meaningful model comparisons, keep the following consistent across runs:

- Dataset split or fold configuration
- Image preprocessing and augmentation
- Number of epochs and stopping criteria
- Evaluation tolerance
- Hardware and software versions

## Roadmap

Planned or likely improvements include:

- More model architectures and pretrained backbones
- More systematic hyperparameter management
- Stronger experiment logging and result aggregation
- Automated comparison tables and plots
- More complete environment/dependency documentation
- Additional tests and validation of edge cases
- Better support for larger-scale experiment sweeps

## License

No license has been selected for this repository yet. Unless a license is added, the code should be treated as **all rights reserved**.
