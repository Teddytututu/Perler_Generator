# Anime Segmentation Dataset Source Record

## Source

- Dataset: Anime Segmentation `imgs-masks`
- Maintainer: SkyTNT
- Project: <https://github.com/SkyTNT/anime-segmentation>
- Dataset repository: <https://huggingface.co/datasets/skytnt/anime-segmentation>
- Downloaded file: `data/imgs-masks.zip`
- Download transport used locally: <https://hf-mirror.com/datasets/skytnt/anime-segmentation>
- Archive size: 182,267,346 bytes
- Archive SHA-256: `e4c8abf4a978c7cabb48eed5be9d6dfeab6b4fe070705713ae20216e77d7211d`
- Verified contents: 1,111 images and 1,111 same-stem masks
- Project repository license: Apache-2.0

The upstream README says this dataset combines data from AniSeg and
`character_bg_seg_data`, then filters and manually cleans the masks so that
they represent anime characters.

## Intended Use

Use the masks to derive high-resolution subject silhouettes and outer-contour
targets. This is the closest public training supervision in the current data
collection to the stage-1 anime portrait task.

This dataset does not label internal structural lines, facial key-feature
lines, contour type, endpoint direction, or artist-designed 48 x 48 bead-grid
outlines. Those labels still require a separate annotation pass.

## Rights Note

The Apache-2.0 license covers the upstream project repository. Do not assume
that it grants commercial rights to every underlying illustration. Keep this
copy for research and prototyping unless a separate review confirms the rights
of the source images and derived annotations.

## Local Layout

```text
data/raw/anime_character_mask/
|-- imgs-masks.zip
|-- SOURCE.md
`-- extracted/
    |-- imgs/       # 1,111 JPG input images
    `-- masks/      # 1,111 same-stem JPG foreground masks
```

The older `anime-character-mask-v1.zip`, `remaining-*.bin`, and
`archive-tail-8mb.bin` files are failed partial-download artifacts. They do not
match the official archive hash and must not be used for training.
