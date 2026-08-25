# VulkanOCR

PaddleOCR's models, running on ncnn over Vulkan, on any GPU with a Vulkan
driver — including AMD cards, where PaddlePaddle itself has no backend at all.

Upstream PaddleOCR is CUDA or CPU. On this machine's Radeon RX 6600 XT
`paddle.device.is_compiled_with_cuda()` and `is_compiled_with_rocm()` are both
false, so PaddleOCR runs on the processor and nothing else. VulkanOCR runs the
same PP-OCRv6 graphs on the GPU through ncnn, at the same accuracy:

| | VulkanOCR (GPU) | PaddleOCR (CPU) |
| --- | --- | --- |
| character error rate | 0.0154 | 0.0154 |
| word error rate | **0.0379** | 0.0498 |
| images read perfectly | 75 % | 73 % — 41 vs 40 of 55, equivalent |
| median page | **97 ms** (67 ms with fp16) | 184 ms |
| CPU time per dense page | **~1.4 s** | ~12.0 s |

Same PP-OCRv6_medium models on both sides. The PaddleOCR column is
`paddleocr 3.7.0` on `paddlepaddle 3.2.2` with oneDNN on — its best CPU
configuration; on paddlepaddle 3.3 the PIR→oneDNN converter refuses every
PP-OCR graph, which is why the `paddle` extra pins `<3.3`. Every row is
produced by a committed runner and `benchmarks/compare_engines.py`, from one
generation of the corpus.

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
  engine.py         facade: load once, read(rgb) -> OcrResult
  catalog.py        model sets as data: PP-OCRv6 tiny/small/medium, PP-OCRv5 mobile
  cli.py            the `vulkanocr` command (`--all-gpus` pools every device)
  parallel.py       one engine process per GPU; a slow card can help, never hurt
  proof.py          sysfs gpu_busy_percent sampling, as a context manager
tests/              the suite; the live tests skip without a GPU
benchmarks/         corpus, scorer, shared measurement loop, one runner per engine, the batching PoCs
docs/               benchmarks, engine notes, and the findings of the first pass
samples/            the images the README and tests quote
```

About fourteen hundred lines of engine and as much again in tests — sizes
that drift, so the claim is the shape, not a census. Dependencies are `ncnn`, `numpy` and
`opencv` — no PaddlePaddle, no ONNX, no polygon clipper.

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
git clone https://github.com/Avafly/PaddleOCR-ncnn-CPP   # PP-OCRv6, MIT
git clone https://github.com/nihui/ncnn-android-ppocrv5 nihui-port  # PP-OCRv5, BSD-3
# cloned elsewhere? point VULKANOCR_MODELS_ROOT at the directory holding both
```

**Optional**, by what you want to do:

| you want to | install |
| --- | --- |
| run development gates | `.venv/bin/python -m pip install -e '.[dev]'` (pytest, Ruff, Pyright) |
| regenerate the corpus | `.venv/bin/python -m pip install -e '.[corpus]'` (Pillow, used by `benchmarks/make_corpus.py`) |
| run the PaddleOCR comparison | the `paddle` extra — in a **separate** venv, never this one: `python3 -m venv ~/paddle-venv && ~/paddle-venv/bin/python -m pip install 'vulkanocr[paddle] @ file://'$PWD` ([docs/benchmarks.md](docs/benchmarks.md) says why, and why it pins paddlepaddle `<3.3`) |
| run the Tesseract comparison | the system binary: `sudo apt install tesseract-ocr` (Debian/Ubuntu) |

## Run it

```sh
.venv/bin/vulkanocr samples/sample-applet.png                # PP-OCRv6 medium
.venv/bin/vulkanocr samples/sample-applet.png --models v6-tiny
.venv/bin/vulkanocr samples/sample-applet.png --precision fp16
.venv/bin/vulkanocr samples/sample-applet.png --precision int8  # quantized graphs
.venv/bin/python -m pytest -q                                # live tests skip without a GPU
.venv/bin/ruff check .
.venv/bin/pyright
```

`--precision fp32|fp16|int8` states every ncnn precision option explicitly.
`int8` enables quantized arithmetic; it does not quantize a float graph, so use
it with a quantized model profile.

The command prints the device it chose, the lines with their coordinates and
confidence, and attributable GPU telemetry while it works.

| CLI telemetry state | Scope | Meaning |
| --- | --- | --- |
| `drm-fdinfo, process-wide` | This VulkanOCR process | Preferred on AMD and Intel when DRM engine counters exist. |
| `amd-gpu-busy-percent, system-wide` | Whole selected AMD GPU | Fallback; activity may include other processes. |
| `GPU telemetry unavailable: ... no engine counters ...` | None | Driver, including NVIDIA configurations without DRM engine accounting, exposes no supported process counter. |
| `GPU telemetry unavailable: ... does not match ...` | None | Available telemetry belongs to another or ambiguously identical device and is not accepted as proof. |

A supported provider can report zero activity; that differs from unsupported
telemetry. Exact provider behavior and benchmark evidence live in
[the benchmark notes](docs/benchmarks.md).

## Accuracy notes

The port originally followed nihui's reference exactly, including its flat
1.95× box enlargement in place of DB's unclip. That costs real accuracy on
small text — accents and last glyphs shaved off the crop — and
`src/vulkanocr/detection.py` now applies DB's own rule, `area × ratio / perimeter`, which
for a `minAreaRect` is one expression and needs no clipper. It closed the gap
to upstream (CER 0.0223 → 0.0156) and made reading *faster*, because a taller
box yields a narrower 48-px crop.

Active gaps are tracked in [TODO.md](TODO.md), including occasional dropped
spaces and the lack of an approved Hebrew model.

Known icon readings can be filtered without banning CJK globally by supplying
both an opt-in policy and page context:

```python
from vulkanocr import FalsePositivePolicy, OcrEngine, RecognitionContext

engine = OcrEngine(
    models,
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
