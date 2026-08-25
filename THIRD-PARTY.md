# Third-party work this project stands on

Nothing here is a fork. VulkanOCR is an original engine that loads models
other people converted, and follows pre- and post-processing from the named
reference implementations. Each runtime/model dependency is named below with
its licence, so a reader can see exactly what is borrowed and from whom.

| What | Who | Licence | Used how |
| --- | --- | --- | --- |
| [PP-OCRv5/v6 and PP-LCNet text-line orientation weights](https://github.com/PaddlePaddle/PaddleOCR) | PaddlePaddle / PaddleOCR | Apache-2.0 | the catalogued models themselves |
| [ncnn ports of PP-OCRv6 and PP-LCNet](https://github.com/Avafly/PaddleOCR-ncnn-CPP/releases/tag/v0.3.0) (`PaddleOCR-ncnn-CPP` v0.3.0) | Avafly | MIT | `.param`/`.bin` graphs and `angle_net.cpp` as the orientation preprocessing reference, fetched at setup |
| [ncnn port of PP-OCRv5](https://github.com/nihui/ncnn-android-ppocrv5/releases/tag/20260527.671ac4a) (`ncnn-android-ppocrv5` 20260527.671ac4a) | nihui (ncnn's author) | BSD 3-Clause | `.param`/`.bin` graphs, and `ppocrv5.cpp` as the reference for detection and CTC decode |
| [ncnn](https://github.com/Tencent/ncnn) | Tencent | BSD 3-Clause | the inference runtime, from PyPI |
| NumPy | NumPy developers | BSD 3-Clause | image/tensor arrays and numeric geometry |
| OpenCV (`opencv-python-headless`) | OpenCV team / wheel maintainers | Apache-2.0 | image I/O, contours, geometry, resizing and affine transforms |

The two model repositories are **not** vendored here: they are clones of
somebody else's work, and republishing them under this name would misrepresent
authorship and duplicate their history. The README's setup section says where
to fetch them (and `VULKANOCR_MODELS_ROOT` names where they live); the weights
are large binaries that belong with their publishers, not in this history.
The catalog and published measurements were verified against the releases
linked in the table. The setup clones currently follow upstream default
branches rather than immutable revisions; `VOCR-0129` tracks the blocked work
for checksum-pinned, per-component installation.

The detection path in `src/vulkanocr/detection.py` starts from nihui's
`ppocrv5.cpp`, with corrections recorded beside the code and pinned by tests:
it evaluates every contour rather than truncating at 1,000, maps each axis to
source coordinates before fitting the rotated rectangle, enforces the
short-side orientation invariant, and replaces the flat 1.95 enlargement with
DB's own unclip rule. The last change measurably recovers the accuracy the flat
rule loses on small text. `src/vulkanocr/orientation.py` follows Avafly's
PP-LCNet smart-resize, normalization and two-class decode; it applies each
line's decision rather than Avafly's optional page-majority vote.
