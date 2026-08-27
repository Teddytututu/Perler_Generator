# COIL-20 Dataset Source Record

## Source

- Dataset: Columbia Object Image Library (COIL-20), processed version
- Publisher: Columbia University CAVE Laboratory
- Official page: <https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php>
- Download URL: <https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip>
- Archive size: 12,684,276 bytes
- Archive SHA-256: `517c5594820eb40066ba0ff6842e7f09392bf7fde849bf9cb9c28445b0f29e88`
- Last modified according to the official server: 2002-07-24

## Contents

- 20 household and everyday objects
- 72 views per object, captured at 5-degree rotation intervals
- 1,440 processed PNG images in total
- Grayscale images with the background removed
- Image size: 128 x 128 pixels
- Filename pattern: `obj{object_id}__{angle}.png`

## Citation

S. A. Nene, S. K. Nayar, and H. Murase, "Columbia Object Image Library
(COIL-20)," Technical Report CUCS-005-96, Columbia University, February 1996.

## Usage Notes

COIL-20 is useful for small-scale experiments involving object recognition,
clustering, feature extraction, image retrieval, and multi-view consistency.
Its black background and controlled turntable views are less representative of
natural photographs, so results should not be treated as evidence of performance
on cluttered real-world scenes.

The official download page does not state a software-style redistribution
license. Treat the dataset as research data, preserve the citation, and review
usage rights separately before commercial redistribution.

## Local Layout

```text
data/raw/coil20/
|-- coil-20-proc.zip
|-- extracted/
|   `-- coil-20-proc/
|       `-- obj{object_id}__{angle}.png
`-- SOURCE.md
```
