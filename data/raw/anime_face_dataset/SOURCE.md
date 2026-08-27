# Anime Face Dataset Source Record

## Source

- Dataset: Anime Face Dataset
- Kaggle reference: `splcher/animefacedataset`
- Kaggle URL: <https://www.kaggle.com/datasets/splcher/animefacedataset>
- Dataset version: 3
- Last updated: 2019-10-13
- Archive size: 413,825,766 bytes
- Extracted image count: 63,565
- Archive MD5: `a0558aaf6da58b037e52e9910fa7e052`
- Kaggle license field: `Database: Open Database, Contents: Database Contents`
- Upstream description: images scraped from `www.getchu.com` and cropped as anime faces

## Intended Use

This copy is intended for algorithm research and prototype experiments on anime-face priors, data loading, preprocessing, and contour-model feasibility.

Do not assume that the Kaggle database license grants commercial rights to every underlying illustration. Before using the images in a commercial product or redistributing them, perform a separate legal review of the original image rights and the database/content licenses.

## Technical Limitations

The dataset contains already-cropped square anime faces, but many source images are very small. Initial inspection found examples between roughly 46 and 93 pixels per side. Upscaling these images does not create real contour detail.

Recommended usage:

- suitable for anime-face representation pretraining;
- suitable for testing data pipelines and model interfaces;
- suitable for weakly supervised or self-supervised experiments;
- not sufficient as the only source of high-quality pixel-accurate contour labels;
- not sufficient as the gold-standard evaluation set.

For the final outline model, add a smaller, high-resolution, manually reviewed dataset with manually designed high-resolution and bead-grid contour labels.

## Local Layout

```text
data/raw/anime_face_dataset/
├─ animefacedataset-v3.zip
├─ extracted/
│  └─ images/
└─ SOURCE.md
```

Generated audit reports and processed datasets are intentionally excluded from source control.

