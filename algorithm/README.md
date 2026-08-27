# Stage-1 Outline Implementation

This directory implements the first-stage semantic outline pipeline described
in `../STAGE1_OUTLINE_IMPLEMENTATION.md`.

## Module Map

The design calls for the following Python modules. They are implemented in the
documented version order instead of being created as empty placeholders.

| Version | Modules | Purpose |
| --- | --- | --- |
| v0.0 | `data/dataset.py`, `data/rasterize.py`, `data/validators.py` | Stable sample format and validation |
| v0.0 | `metrics/closure.py`, `continuity.py`, `smoothness.py`, `pixel_metrics.py` | Reproducible acceptance metrics |
| v0.0 | `visualization/overlay.py`, `report.py` | Dataset and prediction inspection |
| v0.1 | `baselines/canny.py`, `mask_boundary.py`, `learned_edge.py` | Traditional comparison baselines |
| v0.2 | `models/encoder.py`, `decoder.py`, `heads.py`, `outline_model.py` | High-resolution semantic outline network |
| v0.2 | `losses/classification.py`, `total.py` | Subject, binary outline, and type losses |
| v0.3 | `models/grid_projection.py` | Learned 48 x 48 outline projection |
| v0.4 | `losses/topology.py`, `smoothness.py` | clDice and local shape constraints |
| v0.5 | `postprocess/skeletonize.py`, `endpoint.py`, `gap_repair.py`, `cleanup.py` | Constrained local repair |
| v0.2+ | `training/trainer.py`, `callbacks.py` | Reproducible training loop |
| v0.2+ | `inference/predictor.py`, `serializer.py` | Stable inference contract |

The current implementation covers v0.0, the deterministic portion of v0.1,
and the first trainable v0.2 bootstrap loop. The learned-edge baseline still
needs a reviewed HED or PiDiNet checkpoint.

## Commands

Build bootstrap samples from the downloaded anime image/mask pairs:

```powershell
python algorithm/scripts/build_outline_dataset.py `
  --images data/raw/anime_character_mask/extracted/imgs `
  --masks data/raw/anime_character_mask/extracted/masks `
  --output data/processed/anime_outline_bootstrap_v1
```

The generated 48 x 48 labels are automatic bootstrap labels, not final artist
ground truth. Every generated sample is marked `needs_review`.

Validate a standardized dataset:

```powershell
python algorithm/scripts/validate_dataset.py `
  --dataset data/processed/anime_outline_bootstrap_v1 `
  --output data/reports/data_quality/outline_dataset_validation/anime_outline_bootstrap_v1
```

Evaluate Canny and subject-mask boundary baselines:

```powershell
python algorithm/scripts/evaluate_baselines.py `
  --dataset data/processed/anime_outline_bootstrap_smoke `
  --output data/reports/outline_generation/baselines/anime_outline_baselines_smoke `
  --methods canny mask_boundary
```

`OutlineModel` currently uses `TinyHierarchicalEncoder`, a dependency-free
four-scale encoder with the same feature contract expected from SegFormer. It
is intentionally untrained. Replacing the encoder must not change the output
dictionary or the high-resolution/grid tensor shapes.

Run tests without installing the package:

```powershell
$env:PYTHONPATH = "algorithm/src"
pytest algorithm/tests
```

Create a reproducible split and start bootstrap training:

```powershell
python algorithm/scripts/create_dataset_split.py `
  --dataset data/processed/anime_outline_bootstrap_v1 `
  --output data/splits/anime_outline_bootstrap_v1.json

python algorithm/scripts/train_outline.py `
  --dataset data/processed/anime_outline_bootstrap_v1 `
  --split-manifest data/splits/anime_outline_bootstrap_v1.json `
  --output data/checkpoints/outline-bootstrap-v1
```

Bootstrap labels supervise outer contours only. Structural and key-feature
classes require reviewed artist labels before they can be learned.

For the ordinary-image path, fine-tune the pretrained person segmenter and run
inference without supplying a mask:

```powershell
python algorithm/scripts/train_person_segmenter.py `
  --dataset data/processed/anime_outline_bootstrap_v1 `
  --split-manifest data/splits/anime_outline_bootstrap_v1.json `
  --output data/checkpoints/person-segmenter-v1

python algorithm/scripts/predict_person_outline.py `
  --input path/to/image.jpg `
  --checkpoint data/checkpoints/person-segmenter-v1/best.pt `
  --output data/inference/example
```

The explicit image-to-mask training entry point uses the existing masks only as
temporary supervision. The model receives RGB images only, verifies that the
train/validation/test splits cover every dataset sample without overlap, then
calibrates an inference threshold on the validation split. The prediction
script reads that threshold directly from the resulting checkpoint:

```powershell
python algorithm/scripts/train_image_to_mask.py `
  --dataset data/processed/anime_outline_bootstrap_v1 `
  --split-manifest data/splits/anime_outline_bootstrap_v1.json `
  --output data/checkpoints/image-to-mask-v1
```

Export predicted masks for every dataset image without opening the legacy mask
labels:

```powershell
python algorithm/scripts/export_image_masks.py `
  --dataset data/processed/anime_outline_bootstrap_v1 `
  --checkpoint data/checkpoints/image-to-mask-v1/best.pt `
  --output data/inference/image-to-mask-v1-all
```

On a CPU-only machine, add `--freeze-backbone --input-size 256 --batch-size 8`
for the initial classifier adaptation. A later GPU run should unfreeze the
backbone for full fine-tuning.

Inference writes the person probability, cleaned mask, high-resolution outline,
48 x 48 grid outline, and an overlay on the original image.
The default foreground threshold is `0.05` for the unmodified torchvision
checkpoint and `0.8` when a fine-tuned checkpoint is supplied. Both values
were calibrated on the bootstrap validation split.

Evaluate the untouched test split before or after fine-tuning:

```powershell
python algorithm/scripts/evaluate_person_segmenter.py `
  --dataset data/processed/anime_outline_bootstrap_v1 `
  --split-manifest data/splits/anime_outline_bootstrap_v1.json `
  --output data/reports/mask_generation/pretrained_baseline/person-segmenter-pretrained
```
