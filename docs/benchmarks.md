# Benchmarks — how the published numbers were taken (2026-08-21)

Everything here is produced by committed runners in [`benchmarks/`](../benchmarks)
against the engine as it stands, in one generation of the corpus. Nothing in
this file is hand-run: if a number cannot be reproduced by a script, it is not
here.

## The corpus

`benchmarks/make_corpus.py` renders 5 texts × 11 degradations = 55 images with
exact ground truth: three sizes, three fonts, 5° and 12° skew, blur, noise,
JPEG q30, and a faded scan. Synthetic on purpose — a comparison needs
identical inputs and exact truth for every engine. A hand-transcribed scan
holdout is still the honest acceptance corpus; this is not it and says so.

`benchmarks/scoring.py` scores a read: CER and WER as total edit distance over
total length (never a mean of per-image rates), whitespace-normalised, NFC,
reading order ignored. It is unit-tested in `tests/test_scoring.py`.

```sh
.venv/bin/python benchmarks/make_corpus.py /tmp/corpus
.venv/bin/python benchmarks/read_with_vulkanocr.py /tmp/corpus v6-medium
.venv/bin/python benchmarks/read_with_vulkanocr.py /tmp/corpus v6-medium --fp16
python3 benchmarks/read_with_tesseract.py /tmp/corpus 6
<paddle-venv>/bin/python benchmarks/read_with_paddleocr.py /tmp/corpus
python3 benchmarks/compare_engines.py /tmp
```

The Paddle virtualenv installs the `bench` extra's pins: `paddlepaddle>=3.2,<3.3`.
On 3.3.1 the PIR→oneDNN instruction converter fails on a `conv2d` attribute
(`ConvertPirAttribute2RuntimeAttribute not support`), for every PP-OCR graph,
with no flag or blocklist that avoids it — bisected to the op. `--no-mkldnn`
reproduces that crippled configuration; the default is upstream's fair fight.

## Results — all engines, same 55 images, same scorer

| engine | device | CER | WER | exact | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| vulkanocr/v6-medium | RX 6600 XT | 0.0154 | **0.0379** | **75 %** | 97 ms | 124 ms |
| vulkanocr/v6-medium+fp16 | RX 6600 XT | 0.0154 | **0.0379** | **75 %** | **67 ms** | 86 ms |
| paddleocr 3.7.0 / paddle 3.2.2 / oneDNN | CPU | 0.0154 | 0.0498 | 73 % | 184 ms | 223 ms |
| vulkanocr/v6-tiny | RX 6600 XT | 0.0163 | 0.0660 | 60 % | 22 ms | 35 ms |
| vulkanocr/v5-mobile | RX 6600 XT | 0.0405 | 0.1104 | 51 % | 45 ms | 76 ms |
| tesseract 5.3.4 psm6 | CPU | 0.0459 | 0.1255 | 58 % | 93 ms | 107 ms |

Same PP-OCRv6_medium det+rec graphs behind the first three rows, so those
rows isolate the port and the backend: character accuracy is identical, word
accuracy slightly better here (segmentation), and the GPU is ~2× faster wall
clock at fp32, ~2.7× at fp16, which measured no accuracy cost (CER identical
to the fourth decimal).

Per-degradation CER, from the same run:

| variant | v6-medium | +fp16 | paddle+oneDNN | tesseract |
| --- | --- | --- | --- | --- |
| clean 28px sans | 0.006 | 0.006 | 0.006 | 0.011 |
| clean 12px sans | 0.006 | 0.006 | 0.006 | 0.027 |
| blur 5px | 0.000 | 0.000 | 0.000 | 0.011 |
| noise σ25 | 0.000 | 0.000 | 0.002 | 0.011 |
| JPEG q30 | 0.006 | 0.006 | 0.004 | 0.011 |
| faded 40 % | 0.002 | 0.002 | 0.006 | 0.011 |
| skew 5° | 0.000 | 0.000 | 0.000 | 0.032 |
| skew 12° | 0.143 | 0.143 | 0.131 | 0.335 |

## CPU cost, dense real page

`../samples/sample-applet.png`, 585×770 of 12 px UI text: vulkanocr ~671 ms
wall / ~1.4 s CPU; PaddleOCR + oneDNN ~1.2 s wall / ~12.0 s CPU; with oneDNN
off (the 3.3 regression's configuration) ~9-10 s wall / ~90-100 s CPU.
PaddleOCR cannot use this GPU at all: `is_compiled_with_cuda()` and
`is_compiled_with_rocm()` are both false, and Paddle has no Vulkan backend.

## That it really is Vulkan

`benchmarks/gpu_proof.py` reads this process's own amdgpu fdinfo counters:
~1.2-1.3 s of `drm-engine-compute` per ~700 ms read, ~721 MiB VRAM held.
`benchmarks/vulkan_vs_cpu.py` runs the same models with the Vulkan knob off:
identical output, GPU ~1.6× faster for v6-medium. `benchmarks/phase_timings.py`
splits a read: ~38 ms detection, ~1 ms cropping, ~600 ms recognition —
46 sequential net calls, which is where the batching question came from.

## Batching: measured, negative

Three attempts, all in [`benchmarks/`](../benchmarks): concurrent extractors
(0.9-1.0×, the binding serialises), packing every crop into one strip
(≤1.35×, and it corrupts lines — the encoder mixes across the strip), packing
only narrow crops (inside the noise). `dispatch_vs_compute.py` explains it:
per-call overhead ~5 ms, but wide crops are compute-bound at ~900 ms/MPix
fp32. What actually pays is fp16 (1.5× on recognition, no measured accuracy
cost) and the model tier (v6-tiny is 4.5× v6-medium at +0.001 CER).
