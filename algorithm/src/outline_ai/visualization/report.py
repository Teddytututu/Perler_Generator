"""Small contact-sheet builder used by validation reports."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PIL import Image, ImageDraw


def make_contact_sheet(
    images: Sequence[Image.Image],
    labels: Sequence[str] | None = None,
    columns: int = 4,
    cell_size: tuple[int, int] = (256, 284),
) -> Image.Image:
    if columns <= 0:
        raise ValueError("columns must be positive")
    labels = labels or [""] * len(images)
    if len(labels) != len(images):
        raise ValueError("labels and images must have the same length")
    rows = max(1, math.ceil(len(images) / columns))
    sheet = Image.new("RGB", (columns * cell_size[0], rows * cell_size[1]), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        row, column = divmod(index, columns)
        thumb = image.convert("RGB").copy()
        thumb.thumbnail((cell_size[0] - 8, cell_size[1] - 28), Image.Resampling.LANCZOS)
        left = column * cell_size[0] + (cell_size[0] - thumb.width) // 2
        top = row * cell_size[1] + 4
        sheet.paste(thumb, (left, top))
        draw.text((column * cell_size[0] + 4, row * cell_size[1] + cell_size[1] - 20), label, fill="black")
    return sheet

