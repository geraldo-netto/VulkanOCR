# Benchmarks — how the published numbers were taken (2026-08-25)

The main comparison is produced by committed runners in
[`benchmarks/`](../benchmarks) against the current engine, in one generation of
the corpus. The Vulkan rows were rerun after text-line orientation and
whitespace reconstruction landed. Sections explicitly dated 2026-08-21 retain
earlier diagnostic experiments; they are evidence for design choices, not
current headline timings.

## The corpus

`benchmarks/make_corpus.py` renders 5 texts × 11 degradations = 55 images with
exact ground truth: three sizes, three fonts, 5° and 12° skew, blur, noise,
JPEG q30, and a faded scan. Synthetic on purpose — a comparison needs
identical inputs and exact truth for every engine. A hand-transcribed scan
holdout is still the honest acceptance corpus; this is not it and says so.
`ground-truth.json` uses corpus schema version 4 and records each case's
script, BCP 47 language, exact line sequence, font source/version/licence,
palette, rendered size, background objects, variant, and relative image path.
Generation checks each font covers its case's characters and fails before
writing an invalid manifest. Use repeatable `--font-root PATH` options to
search local font collections before the documented system paths.
Positive CJK confusables and non-text glyph/background cases carry explicit
`content_label` and `known_false_readings` fields so false-positive policy is
tested without treating legitimate `花` or `回` as noise.
Each case also declares VulkanOCR, PaddleOCR, and Tesseract model/language
selection. Runners warm and dispatch per declared selection and refuse the
whole corpus if any case has no model for that engine; they never substitute a
convenient default silently. Result case ids are checked against the manifest
before the document is written and again before comparison.

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
.venv/bin/python benchmarks/make_corpus.py --script-samples samples/corpus
.venv/bin/python benchmarks/make_corpus.py --orientation-samples samples/orientation-corpus
.venv/bin/python benchmarks/read_with_vulkanocr.py /tmp/corpus v6-medium
.venv/bin/python benchmarks/read_with_vulkanocr.py /tmp/corpus v6-medium --precision fp16
python3 benchmarks/read_with_tesseract.py /tmp/corpus 6
<paddle-venv>/bin/python benchmarks/read_with_paddleocr.py /tmp/corpus
python3 benchmarks/compare_engines.py /tmp /tmp/corpus
```

The Paddle virtualenv installs the `paddle` extra's pins
(`paddleocr>=3.7,<4`, `paddlepaddle>=3.2,<3.3`); the corpus generator needs
the `corpus` extra (Pillow, FontTools and JSON Schema) in the project's own
venv, and the Tesseract runner needs the system `tesseract-ocr` binary. The
README's Install section lists all of it in one place.
The runner also accepts `--precision int8`, but no quantized profile is
catalogued; enabling int8 flags on the current float profiles is not a
validated int8 benchmark and is therefore absent from the published table.
On 3.3.1 the PIR→oneDNN instruction converter fails on a `conv2d` attribute
(`ConvertPirAttribute2RuntimeAttribute not support`), for every PP-OCR graph,
with no flag or blocklist that avoids it — bisected to the op. `--no-mkldnn`
reproduces that crippled configuration; the default is upstream's fair fight.

## Results — all engines, same 55 images, same scorer

| engine | device | CER | WER | exact | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| vulkanocr/v6-medium | RX 6600 XT | 0.0159 | **0.0390** | 75 % | 99 ms | 118 ms |
| vulkanocr/v6-medium+fp16 | RX 6600 XT | 0.0159 | **0.0390** | 75 % | **66 ms** | 79 ms |
| paddleocr 3.7.0 / paddle 3.2.2 / oneDNN | CPU | 0.0158 | 0.0498 | 73 % | 189 ms | 230 ms |
| vulkanocr/v6-small | RX 6600 XT | 0.0181 | **0.0368** | 73 % | 47 ms | 66 ms |
| vulkanocr/v6-tiny | RX 6600 XT | 0.0163 | 0.0639 | 60 % | 26 ms | 32 ms |
| vulkanocr/v5-mobile | RX 6600 XT | 0.0351 | 0.1017 | 55 % | 45 ms | 80 ms |
| tesseract 5.3.4 psm6 | CPU | 0.0337 | 0.0887 | 73 % | 94 ms | 110 ms |

The same PP-OCRv6 medium detector/recognizer tier underpins the first three
rows: PaddleOCR runs the original graphs while VulkanOCR runs the ncnn
conversion and its catalogued text-line orientation graph. Character accuracy
is equivalent at the precision this corpus can support, word accuracy is
slightly better here (segmentation), and the GPU is ~1.9× faster wall clock at
fp32 and ~2.9× at fp16. fp16 measured no accuracy cost in this run.

The tier rows use copies of the same generated images and truth whose
manifests declare the corresponding VulkanOCR profile. This is required by
the runner: a positional model override that differs from the manifest is a
refusal, not a silent substitution.

The exact-match rows are 41 vs 40 images of 55 — one image, inside the noise
of a corpus this size, so exactness reads as equivalent rather than a win.
The three tie-breaking images are instructive, though: PaddleOCR's two misses
are classic confusions (`O gato` as `0 gato` in a monospace face, an
underscore lost to σ25 noise), ours is two hallucinated Portuguese accents at
16 px (`subíu`, `fícou`) — the thin-stroke weakness the unclip change shrank
but did not eliminate.

Per-degradation CER, from the same run:

| variant | v6-medium | +fp16 | paddle+oneDNN | tesseract |
| --- | --- | --- | --- | --- |
| clean 28px sans | 0.006 | 0.006 | 0.006 | 0.000 |
| clean 12px sans | 0.006 | 0.006 | 0.006 | 0.012 |
| blur 5px | 0.000 | 0.000 | 0.000 | 0.000 |
| noise σ25 | 0.000 | 0.000 | 0.002 | 0.000 |
| JPEG q30 | 0.006 | 0.006 | 0.004 | 0.000 |
| faded 40 % | 0.002 | 0.002 | 0.006 | 0.000 |
| skew 5° | 0.000 | 0.000 | 0.000 | 0.025 |
| skew 12° | 0.146 | 0.146 | 0.135 | 0.320 |

## CPU cost, dense real page (2026-08-21)

`../samples/sample-applet.png`, 585×770 of 12 px UI text: vulkanocr ~671 ms
wall / ~1.4 s CPU; PaddleOCR + oneDNN ~1.2 s wall / ~12.0 s CPU; with oneDNN
off (the 3.3 regression's configuration) ~9-10 s wall / ~90-100 s CPU.
The compared PaddlePaddle build cannot use this GPU:
`is_compiled_with_cuda()` and `is_compiled_with_rocm()` are both false, and it
has no Vulkan backend.

## That it really is Vulkan

The CLI matches this process's DRM fdinfo engine counters to the selected
device when the driver supplies them; this is preferred proof because it
excludes other processes. `benchmarks/gpu_proof.py` is a narrower,
single-engine diagnostic that totals the benchmark process's DRM counters:
~1.2-1.3 s of `drm-engine-compute` per ~700 ms read and ~721 MiB VRAM held in
the 2026-08-21 run. AMD `gpu_busy_percent` remains an explicitly system-wide
CLI fallback when fdinfo lacks engine counters. With `--all-gpus`, fdinfo sees
the parent process's detection and optional orientation work, not recognition
in child workers.
`benchmarks/vulkan_vs_cpu.py` runs the same models with the Vulkan knob off:
identical output, GPU ~1.2× faster for v6-medium on the current dense-page
rerun. `benchmarks/phase_timings.py`
profiles detection, OpenCV cropping, and recognition in isolation: ~38 ms,
~1 ms, and ~594 ms respectively across 46 sequential recognition calls on the
current dense page. The optional orientation pass used by PP-OCRv6 is excluded
from that script; the full CLI read measured 665 ms median over three warm
passes on 2026-08-25.

## The interpolation experiment (2026-08-21, retained negative)

The earlier run's tie-breaking miss (`subíu`/`fría` for 16 px
`subiu`/`fria`) looked like a preprocessing bug: on that single crop, cubic
or Lanczos
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
mis-price the pool for its whole life. The current result on RX 6600 XT +
Radeon 610M is **668 → 617 ms (1.08x), text identical**; on near-equal devices
the same policy can split more of the page. On a machine with one hardware
device the pool builds no worker fleet at all — it is the single engine, same
answers, nothing spawned.

## Batching (2026-08-21, retained negative)

Three attempts, all in [`benchmarks/`](../benchmarks): concurrent extractors
(0.9-1.0×, the binding serialises), packing every crop into one strip
(≤1.35×, and it corrupts lines — the encoder mixes across the strip), packing
only narrow crops (inside the noise). The contemporaneous
`dispatch_vs_compute.py` run measured per-call overhead at ~5 ms and wide
crops at ~900 ms/MPix fp32. The PoCs predate the public `crops()` seam returning
`(region, patch)` pairs and need an unpacking update before they can be rerun;
their retained result is not a current gate. The current corpus still shows
the useful production knobs: fp16 has no measured accuracy cost, and v6-tiny
is ~3.8× v6-medium at +0.0004 CER.
