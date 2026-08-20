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
PP-OCR graph, which is why the `bench` extra pins `<3.3`. Every row is
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
  cli.py            the `vulkanocr` command
tests/              21 tests; the live ones skip without a GPU
benchmarks/         corpus, scorer, one runner per engine, the batching PoCs
docs/               benchmarks, engine notes, and the findings of the first pass
samples/            the images the README and tests quote
```

Under a thousand lines of engine, six hundred of tests — sizes that drift,
so the claim is the shape, not a census. Dependencies are `ncnn`, `numpy` and
`opencv` — no PaddlePaddle, no ONNX, no polygon clipper.

## Run it

The model graphs are third-party ports and are not vendored here; fetch them
once (see [THIRD-PARTY.md](THIRD-PARTY.md)):

```sh
git clone https://github.com/Avafly/PaddleOCR-ncnn-CPP   # PP-OCRv6, MIT
git clone https://github.com/nihui/ncnn-android-ppocrv5 nihui-port  # PP-OCRv5, BSD-3

python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'

.venv/bin/vulkanocr samples/sample-applet.png                # PP-OCRv6 medium
.venv/bin/vulkanocr samples/sample-applet.png --models v6-tiny
.venv/bin/python -m pytest -q                                # 21 tests
```

The command prints the device it chose, the lines with their coordinates and
confidence, and the GPU's `gpu_busy_percent` while it works — so "it ran on
the GPU" is observable rather than asserted. For the same claim measured per
process, `benchmarks/gpu_proof.py` reads this process's own amdgpu counters.

## Accuracy notes

The port originally followed nihui's reference exactly, including its flat
1.95× box enlargement in place of DB's unclip. That costs real accuracy on
small text — accents and last glyphs shaved off the crop — and
`src/vulkanocr/detection.py` now applies DB's own rule, `area × ratio / perimeter`, which
for a `minAreaRect` is one expression and needs no clipper. It closed the gap
to upstream (CER 0.0223 → 0.0156) and made reading *faster*, because a taller
box yields a narrower 48-px crop.

Known gaps, all recorded in [docs/findings.md](docs/findings.md): 90°-rotated text is
unreadable (no orientation classifier), glyph-font icons decode as CJK noise
below ~0.6 confidence, and Hebrew has no pretrained model anywhere in the
Paddle ecosystem.

## Licence

MIT — see [LICENSE](LICENSE). The models and the reference implementation
belong to their authors; [THIRD-PARTY.md](THIRD-PARTY.md) names each one.
