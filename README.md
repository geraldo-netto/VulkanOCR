# VulkanOCR

PaddleOCR's models, running on ncnn over Vulkan, on hardware GPUs exposed by
ncnn's Vulkan runtime — including this AMD card, which the compared
PaddlePaddle build cannot use.

On this machine's Radeon RX 6600 XT
`paddle.device.is_compiled_with_cuda()` and `is_compiled_with_rocm()` are both
false, so PaddleOCR runs on the processor and nothing else. VulkanOCR runs the
corresponding PP-OCRv6 medium detector and recognizer on the GPU through ncnn.
On the synthetic corpus its character accuracy is equivalent, its word error
rate is lower, and it is faster:

| | VulkanOCR (GPU) | PaddleOCR (CPU) |
| --- | --- | --- |
| character error rate | 0.0159 | 0.0158 |
| word error rate | **0.0390** | 0.0498 |
| images read perfectly | 75 % | 73 % — 41 vs 40 of 55, equivalent |
| warm, model-loaded median page | **99 ms** (66 ms with fp16) | 189 ms |

The same PP-OCRv6 medium detector/recognizer tier underpins both sides, though
VulkanOCR uses the converted ncnn graphs and also runs its catalogued
text-line orientation graph. The PaddleOCR column is
`paddleocr 3.7.0` on `paddlepaddle 3.2.2` with oneDNN on — its best CPU
configuration; on paddlepaddle 3.3 the PIR→oneDNN converter refuses every
PP-OCR graph, which is why the `paddle` extra pins `<3.3`. Every row is
produced by a committed runner and `benchmarks/compare_engines.py`, from one
generation of the corpus rerun on 2026-08-25 after orientation correction and
whitespace reconstruction landed.

55 rendered pages with exact ground truth, five texts across eleven
degradations — sizes, fonts, skew, blur, noise, JPEG and a faded scan. The
corpus generator, the scorer and every runner are in [`benchmarks/`](benchmarks), so the
numbers can be reproduced rather than believed. [`docs/benchmarks.md`](docs/benchmarks.md) records how
they were taken and what the corpus does not cover.

## What it is

```
src/vulkanocr/      the engine, installed as the `vulkanocr` package
  device.py         hardware-only Vulkan selection; software rasterisers are refused
  detection.py      DB preprocess -> probability map -> oriented boxes -> unclip
  recognition.py    affine crop -> CTC head -> greedy decode
  orientation.py    correct 0°/180° text-line direction before recognition
  layout.py         infer conservative separators between detector regions
  engine.py         facade: load once, read(rgb) -> OcrResult
  catalog.py        independently composable components and named model profiles
  options.py        explicit fp32/fp16/int8 execution policy
  policy.py         opt-in, page-aware filtering of known false readings
  cli.py            the `vulkanocr` command (`--all-gpus` pools every device)
  parallel.py       detect/orient once; schedule recognition across GPU workers
  scheduling.py     cost-aware crop assignment; slow devices may remain idle
  workers.py        one recognition-only child process per hardware GPU
  proof.py          per-process DRM telemetry with an AMD system-wide fallback
tests/              the suite; the live tests skip without a GPU
benchmarks/         corpus, scorer, shared measurement loop, one runner per engine, the batching PoCs
docs/               benchmarks, engine notes, whitespace contract, and historical findings
samples/            the images the README and tests quote
```

Dependencies are `ncnn`, `numpy` and `opencv` — no PaddlePaddle, no ONNX, no
polygon clipper.

## Install

**Mandatory.** Python ≥ 3.11 and a working Vulkan driver for the GPU (on this
host, Mesa's RADV; `vulkaninfo` should list the card). Without a hardware
Vulkan device the engine refuses to run — there is deliberately no CPU
fallback. Everything Python-side installs with the package itself:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .   # pulls ncnn, numpy, opencv-python-headless
```

The model graphs are third-party ports and are not vendored here; fetch them
once (see [THIRD-PARTY.md](THIRD-PARTY.md)):

```sh
git clone https://github.com/Avafly/PaddleOCR-ncnn-CPP   # ncnn conversions, MIT
git clone https://github.com/nihui/ncnn-android-ppocrv5 nihui-port  # ncnn conversion, BSD-3
# cloned elsewhere? point VULKANOCR_MODELS_ROOT at the directory holding both
```

The catalog and published measurements were verified with Avafly release
[`v0.3.0`](https://github.com/Avafly/PaddleOCR-ncnn-CPP/releases/tag/v0.3.0)
and nihui release
[`20260527.671ac4a`](https://github.com/nihui/ncnn-android-ppocrv5/releases/tag/20260527.671ac4a).
The convenient clone commands above follow the repositories' current default
branches; they are not immutable dependency pins. `VOCR-0129` tracks a
checksum-pinned component installer.

The converted repositories have the licences shown above; the underlying
PaddlePaddle model weights are Apache-2.0. Avafly also supplies the PP-LCNet
text-line orientation graph used by the v6 profiles.

**Optional**, by what you want to do:

| you want to | install |
| --- | --- |
| run development gates | `.venv/bin/python -m pip install -e '.[dev]'` (pytest, Ruff, Pyright) |
| regenerate the corpus | `.venv/bin/python -m pip install -e '.[corpus]'` (Pillow, FontTools and JSON Schema) |
| run the PaddleOCR comparison | the `paddle` extra — in a **separate** venv, never this one: `python3 -m venv ~/paddle-venv && ~/paddle-venv/bin/python -m pip install 'vulkanocr[paddle] @ file://'$PWD` ([docs/benchmarks.md](docs/benchmarks.md) says why, and why it pins paddlepaddle `<3.3`) |
| run the Tesseract comparison | the system binary: `sudo apt install tesseract-ocr` (Debian/Ubuntu) |

## Run it

```sh
.venv/bin/vulkanocr samples/sample-applet.png                # PP-OCRv6 medium
.venv/bin/vulkanocr samples/sample-applet.png --models v6-tiny
.venv/bin/vulkanocr samples/sample-applet.png --precision fp16
.venv/bin/vulkanocr samples/sample-applet.png --repeat 5         # five extra timed reads
.venv/bin/vulkanocr samples/sample-applet.png --all-gpus        # pool hardware GPUs
.venv/bin/python -m pytest -q                                # live tests skip without a GPU
.venv/bin/ruff check .
.venv/bin/pyright
```

`--precision fp32|fp16|int8` states every ncnn precision option explicitly.
`int8` enables quantized arithmetic; it does not quantize a float graph. No
quantized model profile is catalogued yet, so the current CLI model choices do
not provide a validated int8 path.

The command prints the device it chose, the lines with their coordinates and
confidence, and attributable GPU telemetry while it works.

| CLI telemetry state | Scope | Meaning |
| --- | --- | --- |
| `drm-fdinfo, process-wide` | This VulkanOCR process | Preferred on AMD and Intel when DRM engine counters exist. |
| `amd-gpu-busy-percent, system-wide` | Whole selected AMD GPU | Fallback; activity may include other processes. |
| `GPU telemetry unavailable: ... no engine counters ...` | None | Driver, including NVIDIA configurations without DRM engine accounting, exposes no supported process counter. |
| `GPU telemetry unavailable: ... does not match ...` | None | Available telemetry belongs to another or ambiguously identical device and is not accepted as proof. |

A supported provider can report zero activity; that differs from unsupported
telemetry. With `--all-gpus`, fdinfo covers the CLI parent process (detection
and optional orientation), not recognition performed by child workers; the AMD
fallback is system-wide and may include unrelated work. Exact provider
behavior and benchmark evidence live in
[the benchmark notes](docs/benchmarks.md).

## Accuracy notes

The port originally followed nihui's reference exactly, including its flat
1.95× box enlargement in place of DB's unclip. That costs real accuracy on
small text — accents and last glyphs shaved off the crop — and
`src/vulkanocr/detection.py` now applies DB's own rule, `area × ratio / perimeter`, which
for a `minAreaRect` is one expression and needs no clipper. It closed the gap
to upstream (CER 0.0223 → 0.0156) and made reading *faster*, because a taller
box yields a narrower 48-px crop.

PP-OCRv6 profiles include a text-line orientation graph and are exercised at
all four cardinal page rotations. The PP-OCRv5 profile has no orientation
graph. VulkanOCR also reconstructs conservative separators when the detector
splits one visual line into multiple regions; the exact geometry, punctuation
and bidirectional limits are documented in
[docs/whitespace.md](docs/whitespace.md). It cannot recover a space lost inside
one recognition region.

Active gaps are tracked in [TODO.md](TODO.md). The open table is currently
empty; blocked work covers Hebrew model approval and artifacts, validated int8
artifacts, and a checksum-pinned component model store.

Known icon readings can be filtered without banning CJK globally by supplying
both an opt-in policy and page context:

```python
from vulkanocr import FalsePositivePolicy, OcrEngine, RecognitionContext, models_for

engine = OcrEngine(
    models_for(),
    false_positive_policy=FalsePositivePolicy(frozenset({"花", "回"})),
    recognition_context=RecognitionContext(page_languages=frozenset({"en"})),
)
```

Low-confidence `花`/`回` stays valid on a Chinese/Japanese page, without page
context, or when only broad model capability is known. Filtered regions are
reported separately from regions that failed to decode.

## Licence

MIT — see [LICENSE](LICENSE). The models and the reference implementation
belong to their authors; [THIRD-PARTY.md](THIRD-PARTY.md) names each one.
