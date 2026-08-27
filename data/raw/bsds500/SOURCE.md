# BSDS500 Source Record

## Source

- Dataset: Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500)
- Publisher: UC Berkeley Computer Vision Group
- Official resource page: <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html>
- Official archive: `BSR_bsds500.tgz`
- Archive size: 70,763,455 bytes
- Archive SHA-256: `97e49d31764f3912f0c4122707d53062ac9e783ba0f095e447a4d53c1a41af8e`
- Verified split: 200 train, 100 validation, 200 test images
- Verified labels: one MATLAB ground-truth file per image, containing multiple
  human segmentations and boundaries

## Citation

P. Arbelaez, M. Maire, C. Fowlkes, and J. Malik, "Contour Detection and
Hierarchical Image Segmentation," IEEE TPAMI, 33(5), 2011.

The downloaded archive includes the paper and the original benchmark code.

## Intended Use

Use BSDS500 as the standard generic contour-detection benchmark and for
boundary-aware pretraining. Report standard boundary precision/recall metrics
such as ODS, OIS, and AP when the evaluation adapter is implemented.

BSDS500 is not anime-specific and its human boundaries include scene and
texture boundaries that this project may intentionally suppress. It must not
be the only validation set for the bead-outline product target. Keep a separate
anime portrait test set with outer-contour closure, continuity, key-feature
recall, and 48 x 48 grid metrics.

## Rights Note

Treat BSDS500 as research benchmark data. Preserve the citation and review the
official terms before commercial redistribution or use outside research.

## Local Layout

```text
data/raw/bsds500/
|-- BSR_bsds500.tgz
|-- SOURCE.md
`-- extracted/BSR/BSDS500/data/
    |-- images/{train,val,test}/
    `-- groundTruth/{train,val,test}/
```

`BIDS-BSDS500-master.zip` and `BIDS-BSDS500/` are incomplete fallback-download
artifacts and must not be used. The official Berkeley archive above passed the
full structure and sample-count checks.
