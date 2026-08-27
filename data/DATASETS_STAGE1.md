# Stage-1 Outline Dataset Selection

The first outline model needs two different kinds of supervision. No single
public dataset fully matches the product label format.

| Dataset | Local status | Primary role | Directly useful labels | Main limitation |
| --- | --- | --- | --- | --- |
| Anime Segmentation | Downloaded and verified | Anime-domain training | Character foreground mask and derived outer contour | No internal/key-feature/grid outline labels |
| BSDS500 | Downloaded and verified | Generic contour benchmark and boundary pretraining | Multi-annotator human boundaries | Natural images; includes boundaries the product may suppress |
| Anime Face Dataset | Already present | Representation pretraining and pipeline tests | Images only | Small source resolution and no contour ground truth |
| COIL-20 | Already present | Smoke tests for silhouette extraction | Clean object/background separation | Tiny grayscale controlled-object dataset, not anime |

## Recommended First Experiment

1. Derive a one-pixel outer boundary from each Anime Segmentation mask and use
   it for the high-resolution `outer` class.
2. Use BSDS500 only for boundary-aware initialization and standard boundary
   evaluation; do not mix its test set into training.
3. Hand-label 100 to 300 anime portraits with `outer`, `structural`,
   `key_feature`, and artist-designed 48 x 48 grid outlines for the feasibility
   set described in `STAGE1_OUTLINE_IMPLEMENTATION.md`.
4. Split anime data by source/character identity where possible. A random
   image-level split can leak near-duplicate characters across train and test.

## Required Derived Data

Generated boundaries and resized grid labels belong under `data/processed/`,
not beside the immutable raw archives. Record the derivation parameters,
especially mask threshold, boundary connectivity, contour thickness, crop,
resize method, and any morphology used to close gaps.

Public masks provide only an outer-contour bootstrap. They cannot satisfy the
final acceptance criteria for structural lines, key-feature retention, artistic
simplification, or closed 48 x 48 bead-grid outlines without additional manual
annotation.
