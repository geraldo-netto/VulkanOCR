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
`ground-truth.json` uses corpus schema version 1 and records each case's
script, BCP 47 language, exact line sequence, font/licence, palette, rendered
size, background objects, variant, and relative image path. Generation fails
before writing an invalid manifest.

`benchmarks/scoring.py` scores a read: CER and WER as total edit distance over
total length (never a mean of per-image rates), whitespace-normalised, NFC,
reading order ignored. It is unit-tested in `tests/test_scoring.py`.

Accuracy policy: reading order is excluded. Detector traversal is not a text
recognition error; line content and within-line character/word order remain
significant. Result documents still preserve engine order for diagnostics.
After NFC and whitespace normalisation (which drops empty lines), CER and WER
independently choose the minimum-cost one-to-one line alignment using character
or word edit distance. An unmatched line costs all its units; duplicate lines
remain distinct. Aggregate denominators are total ground-truth characters or
words, and exact means equality of the normalised line multisets.

```sh
.venv/bin/python benchmarks/make_corpus.py /tmp/corpus
.venv/bin/python benchmarks/read_with_vulkanocr.py /tmp/corpus v6-medium
.venv/bin/python benchmarks/read_with_vulkanocr.py /tmp/corpus v6-medium --precision fp16
.venv/bin/python benchmarks/read_with_vulkanocr.py /tmp/corpus v6-medium --precision int8
python3 benchmarks/read_with_tesseract.py /tmp/corpus 6
<paddle-venv>/bin/python benchmarks/read_with_paddleocr.py /tmp/corpus
python3 benchmarks/compare_engines.py /tmp /tmp/corpus
```

The Paddle virtualenv installs the `paddle` extra's pins
(`paddleocr>=3.7,<4`, `paddlepaddle>=3.2,<3.3`); the corpus generator needs
only the `corpus` extra (Pillow) in the project's own venv, and the Tesseract
runner needs the system `tesseract-ocr` binary. The README's Install section
lists all of it in one place.
On 3.3.1 the PIR→oneDNN instruction converter fails on a `conv2d` attribute
(`ConvertPirAttribute2RuntimeAttribute not support`), for every PP-OCR graph,
with no flag or blocklist that avoids it — bisected to the op. `--no-mkldnn`
reproduces that crippled configuration; the default is upstream's fair fight.

## Results — all engines, same 55 images, same scorer

| engine | device | CER | WER | exact | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| vulkanocr/v6-medium | RX 6600 XT | 0.0154 | **0.0379** | 75 % | 97 ms | 124 ms |
| vulkanocr/v6-medium+fp16 | RX 6600 XT | 0.0154 | **0.0379** | 75 % | **67 ms** | 86 ms |
| paddleocr 3.7.0 / paddle 3.2.2 / oneDNN | CPU | 0.0154 | 0.0498 | 73 % | 184 ms | 223 ms |
| vulkanocr/v6-tiny | RX 6600 XT | 0.0163 | 0.0660 | 60 % | 22 ms | 35 ms |
| vulkanocr/v5-mobile | RX 6600 XT | 0.0405 | 0.1104 | 51 % | 45 ms | 76 ms |
| tesseract 5.3.4 psm6 | CPU | 0.0459 | 0.1255 | 58 % | 93 ms | 107 ms |

Same PP-OCRv6_medium det+rec graphs behind the first three rows, so those
rows isolate the port and the backend: character accuracy is identical, word
accuracy slightly better here (segmentation), and the GPU is ~2× faster wall
clock at fp32, ~2.7× at fp16, which measured no accuracy cost (CER identical
to the fourth decimal).

The exact-match rows are 41 vs 40 images of 55 — one image, inside the noise
of a corpus this size, so exactness reads as equivalent rather than a win.
The three tie-breaking images are instructive, though: PaddleOCR's two misses
are classic confusions (`O gato` as `0 gato` in a monospace face, an
underscore lost to σ25 noise), ours is two hallucinated Portuguese accents at
16 px (`subíu`, `fría`) — the thin-stroke weakness the unclip change shrank
but did not eliminate.

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

## The hallucinated accent: measured, also negative

The one tie-breaking image we miss (`subíu`/`fría` for 16 px `subiu`/`fria`)
looked like a preprocessing bug: on that single crop, cubic or Lanczos
resampling reads it clean where the fused bilinear warp does not. But every
global alternative regresses the corpus — cubic-on-upscale CER 0.0154→0.0159,
Lanczos 0.0165, and rectifying at native size before a separate resize (the
PaddleOCR reference's two-step shape) 0.0165 with an exact image lost. The
model is near-tied on that crop (0.977 vs 0.970 confidence), and any kernel
that flips it flips more elsewhere. The fused bilinear crop stays; fixing one
image at the corpus's expense would be tuning the benchmark, not the engine.

## Multi-GPU: measured, positive — after two measured failures

`benchmarks/multi_gpu.py`. A thread pool over both of this desk's GPUs ran
**4x slower** than one: the ncnn binding holds the GIL through `extract`
(a Python spin stalls the full 225 ms of an iGPU extract), so threads
serialise. Process-per-device fixed the parallelism; the scheduler then had
to price work, because on a 7.5x-asymmetric pair a slow card that *takes* a
crop the fast card would finish sooner hurts the page. The dispatcher grants
a slow device a cheap crop only while its cumulative commitment stays under
the fast side's projected work; prices are seeded by a start-up probe strip
and refined by every finished crop, so an unrepresentative probe cannot
mis-price the pool for its whole life. Result on RX 6600 XT + Radeon 610M:
**639 → 573 ms (1.12x), text identical** (re-runs land between 1.07x and
1.12x); on near-equal devices the same policy splits the page and approaches
2x. On a machine with one hardware device the pool builds no worker fleet at
all — it is the single engine, same answers, nothing spawned.

## Batching: measured, negative

Three attempts, all in [`benchmarks/`](../benchmarks): concurrent extractors
(0.9-1.0×, the binding serialises), packing every crop into one strip
(≤1.35×, and it corrupts lines — the encoder mixes across the strip), packing
only narrow crops (inside the noise). `dispatch_vs_compute.py` explains it:
per-call overhead ~5 ms, but wide crops are compute-bound at ~900 ms/MPix
fp32. What actually pays is fp16 (1.5× on recognition, no measured accuracy
cost) and the model tier (v6-tiny is 4.5× v6-medium at +0.001 CER).
