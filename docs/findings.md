# OMNI-0351 P0 spike — findings (2026-08-15)

Half-day spike answering one question: does PaddleOCR-on-ncnn work on this host
without touching either upstream? **Yes.** Working end-to-end OCR on the RX 6600
XT in one session, no fork, no patch, no Paddle installed.

## What is in this folder

| Path | What it is |
| --- | --- |
| `nihui-port/` | clone of `github.com/nihui/ncnn-android-ppocrv5` (BSD 3-Clause, Tencent) — PP-OCRv5 mobile+server det/rec `.param`/`.bin`, plus `ppocrv5.cpp`, the reference pre/post-processing |
| `ppocrv5_keys.txt` | 18384 classes extracted from `ppocrv5_dict.h` (18383 entries + ASCII space); blank is class 0, so the rec head's 18385 outputs are fully accounted for |
| `spike_ocr.py` | throwaway Python port of `ppocrv5.cpp` — det → DB threshold → contours → minAreaRect → enlarge → affine crop → rec → CTC greedy decode |
| `.venv/` | `ncnn 1.0.20260526`, numpy, opencv-python-headless, pillow. No paddle, no paddle2onnx, no pnnx |
| `sample-*.png` | synthetic text, applet screenshot, rotations, noise |

## Results

All runs on `[1] AMD Radeon RX 6600 XT (RADV NAVI23)`, Vulkan, fp16 off,
software devices refused by the same rule the service uses.

| Input | Result |
| --- | --- |
| synthetic 900×320, 5 lines | 5/5 lines correct incl. Portuguese accented words. det 15 ms, rec 175 ms |
| applet screenshot 585×770, dense 12 px UI text | 56 lines, essentially all correct incl. `·` separators. det 26 ms, rec 963 ms (57 boxes, sequential) |
| same, server models | 49 lines, det 118 ms, rec 1021 ms — **fewer boxes than mobile**, no accuracy win here |
| rotated 7° and 25° | 5/5 correct — the rotated-rect crop handles skew for free |
| rotated 90° | fails (3 garbage lines) — no orientation classifier in this port |
| gaussian blur + noise | 5/5 correct, minor space loss |

Model load is 0.6–2.1 s; irrelevant given the service caches `Net`s.

## What this changes in the plan

1. **Conversion chain avoided entirely.** The paddle→onnx→pnnx risk that drove
   the whole P0 disappears: nihui (ncnn's author) publishes converted PP-OCRv5
   graphs under BSD-3, and the weights are Apache-2.0 PaddleOCR. Recipe pins the
   ported `.param`/`.bin` by sha256. Provenance is one hop from ncnn upstream,
   not an anonymous port.
2. **No `pyclipper`, no fixed shapes.** The reference replaces DB unclip with a
   1.95 box enlargement, and ncnn takes variable input sizes — det letterboxes to
   a multiple of 32, rec takes 48×W. Both plan items (geometry dep, width
   buckets) are dead. Remaining deps: ncnn, numpy, opencv.
3. **`opencv-python` arrives with the ncnn wheel already**, so the geometry
   dependency is transitive today; declare it explicitly, no new supply chain.
4. **Effort drops.** P1 engine is a ~250-line port of a file we have, with a
   correct reference implementation to diff against.

## New problem the spike found

**Hebrew is not covered.** The PP-OCRv5 dict is CJK + Latin + 11 stray Cyrillic
glyphs. `שלום` cannot be recognised at any confidence. PaddleOCR's v5
multilingual set is latin/arabic/cyrillic/devanagari — `ppocrv5_hebrew_dict.txt`
does not exist; only the older v2-era `hebrew_dict.txt` does. So a Hebrew lane
means the older multilingual rec model, converted ourselves — the conversion
chain returns, for that lane only.

Given `hebrew_installation.py` and the Hebrew paths already in this project, this
is a product decision, not a detail: ship Latin/CJK OCR now and treat Hebrew as
a separate, later, riskier row.

## Other gaps to carry into P1

- **90° text unreadable** — PaddleOCR's angle classifier is a third model; the
  port omits it. Either accept it or convert `PP-LCNet_x1_0_textline_ori`.
- **Icons read as text** — glyph-font icons in the applet screenshot decoded as
  `花`, `回`, `88`. A confidence floor plus a CJK-in-Latin-context filter would
  drop them; needs a rule, not just a threshold.
- **Spaces occasionally dropped** between token groups (`GPU0%|RadeonRX6600XT`).
  Matches upstream behaviour; box-gap re-insertion is a postprocess choice.
- **Recognition is sequential**: 57 crops ≈ 963 ms. The C++ runs crops in
  parallel; our GPU lease is exclusive, so batching crops into one net call (or
  holding the lease per page) is the throughput answer.

## Verdict

Route is viable and cheaper than planned: **pin the nihui BSD-3 ncnn ports**,
port `ppocrv5.cpp` to a small `omnitensor-ocr-engine` library, wrap it as a
provider, plug into `AdapterKind.OCR`. The acceptance-corpus cost (licensed,
hand-transcribed holdout) is unchanged and is still the largest remaining item.
