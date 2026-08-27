"""Derive bootstrap raster labels from foreground masks."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize


OUTER_CLASS = 1
IGNORE_CLASS = 255


def as_binary(array: np.ndarray, threshold: int | float = 0) -> np.ndarray:
    """Convert an image-like array to a two-dimensional boolean mask."""
    value = np.asarray(array)
    if value.ndim == 3:
        value = value[..., :3].max(axis=2)
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {value.shape}")
    return value > threshold


def mask_to_outer_boundary(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Return a one-pixel boundary for the hole-filled primary foreground."""
    binary = as_binary(mask)
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.uint8)

    labels, count = ndimage.label(binary, structure=ndimage.generate_binary_structure(2, 2))
    if count > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        binary = labels == int(sizes.argmax())

    filled = ndimage.binary_fill_holes(binary)
    rank = 2 if connectivity == 8 else 1
    structure = ndimage.generate_binary_structure(2, rank)
    eroded = ndimage.binary_erosion(filled, structure=structure, border_value=0)
    boundary = filled & ~eroded
    return skeletonize(boundary).astype(np.uint8)


def boundary_to_type_map(boundary: np.ndarray, class_id: int = OUTER_CLASS) -> np.ndarray:
    result = np.zeros(np.asarray(boundary).shape, dtype=np.uint8)
    result[as_binary(boundary)] = class_id
    return result


def resize_label(label: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a categorical label with nearest-neighbor interpolation."""
    image = Image.fromarray(np.asarray(label, dtype=np.uint8), mode="L")
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)).copy()


def project_boundary_to_grid(boundary: np.ndarray, grid_size: tuple[int, int] = (48, 48)) -> np.ndarray:
    """Project any covered high-resolution boundary pixel into a grid cell.

    This is an automatic bootstrap label. It preserves thin paths better than
    ordinary image downsampling but does not replace artist-designed grid labels.
    """
    source = as_binary(boundary)
    grid_width, grid_height = grid_size
    if grid_width <= 0 or grid_height <= 0:
        raise ValueError("grid dimensions must be positive")
    output = np.zeros((grid_height, grid_width), dtype=bool)
    rows, cols = np.nonzero(source)
    if rows.size:
        grid_rows = np.minimum(rows * grid_height // source.shape[0], grid_height - 1)
        grid_cols = np.minimum(cols * grid_width // source.shape[1], grid_width - 1)
        output[grid_rows, grid_cols] = True
    return skeletonize(output).astype(np.uint8)


def project_mask_to_grid_boundary(
    mask: np.ndarray,
    grid_size: tuple[int, int] = (48, 48),
    occupancy_threshold: float = 0.5,
) -> np.ndarray:
    """Project a filled subject mask first, then derive its grid boundary.

    This route is preferred for mask-supervised outer contours because a filled
    region survives downsampling more reliably than a one-pixel path.
    """
    if not 0.0 < occupancy_threshold <= 1.0:
        raise ValueError("occupancy_threshold must be in (0, 1]")
    grid_width, grid_height = grid_size
    if grid_width <= 0 or grid_height <= 0:
        raise ValueError("grid dimensions must be positive")
    source = as_binary(mask).astype(np.uint8) * 255
    image = Image.fromarray(source, mode="L")
    occupancy = np.asarray(
        image.resize((grid_width, grid_height), Image.Resampling.BOX), dtype=np.float32
    ) / 255.0
    grid_mask = occupancy >= occupancy_threshold
    return mask_to_outer_boundary(grid_mask)


def compute_direction_labels(boundary: np.ndarray) -> np.ndarray:
    """Assign one of eight local tangent directions, with 255 off-contour."""
    binary = as_binary(boundary)
    labels = np.full(binary.shape, 255, dtype=np.uint8)
    offsets = np.array(
        [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    )
    height, width = binary.shape
    for row, col in zip(*np.nonzero(binary), strict=True):
        neighbors: list[tuple[int, int]] = []
        for index, (dr, dc) in enumerate(offsets):
            rr, cc = row + dr, col + dc
            if 0 <= rr < height and 0 <= cc < width and binary[rr, cc]:
                neighbors.append((index, rr * width + cc))
        if neighbors:
            labels[row, col] = min(neighbors, key=lambda item: item[1])[0]
    return labels
