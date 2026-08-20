# Third-party work this project stands on

Nothing here is a fork. VulkanOCR is an original engine that loads models
other people converted, and follows the pre- and post-processing of a
reference implementation. Each of those is named below with its licence, so a
reader can see exactly what is borrowed and from whom.

| What | Who | Licence | Used how |
| --- | --- | --- | --- |
| PP-OCRv3/v4/v5/v6 weights | PaddlePaddle / PaddleOCR | Apache-2.0 | the models themselves |
| ncnn ports of PP-OCRv6 (`PaddleOCR-ncnn-CPP`) | Avafly | MIT | `.param`/`.bin` graphs, fetched at setup |
| ncnn port of PP-OCRv5 (`ncnn-android-ppocrv5`) | nihui (ncnn's author) | BSD 3-Clause | `.param`/`.bin` graphs, and `ppocrv5.cpp` as the reference for detection and CTC decode |
| ncnn | Tencent | BSD 3-Clause | the inference runtime, from PyPI |

The two model repositories are **not** vendored here: they are clones of
somebody else's work, and republishing them under this name would misrepresent
authorship and duplicate their history. [`docs/benchmarks.md`](docs/benchmarks.md) and the README's setup
section say where to fetch them; the weights are large binaries that belong with their
publishers, not in this history.

The detection geometry in `src/vulkanocr/detection.py` follows nihui's
`ppocrv5.cpp` with one deliberate divergence, recorded in the code: the flat
1.95 box enlargement is replaced by DB's own unclip rule, which measurably
recovers the accuracy the flat rule loses on small text.
