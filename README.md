# VulkanOCR

PaddleOCR's models, running on ncnn over Vulkan, on any GPU with a Vulkan
driver — including AMD cards, where PaddlePaddle itself has no backend at all.

Upstream PaddleOCR is CUDA or CPU. On this machine's Radeon RX 6600 XT
`paddle.device.is_compiled_with_cuda()` and `is_compiled_with_rocm()` are both
false, so PaddleOCR runs on the processor and nothing else. VulkanOCR runs the
same PP-OCRv6 graphs on the GPU through ncnn, at the same accuracy:

| | VulkanOCR (GPU) | PaddleOCR 3.2.2 + oneDNN (CPU) |
| --- | --- | --- |
| character error rate | 0.0156 | 0.0154 |
| word error rate | **0.0400** | 0.0498 |
| images read perfectly | **76 %** | 73 % |
| median page | **66 ms** | 199 ms |
| CPU time per dense page | **~1.4 s** | ~12.0 s |

55 rendered pages with exact ground truth, five texts across eleven
degradations — sizes, fonts, skew, blur, noise, JPEG and a faded scan. The
corpus generator, the scorer and every runner are in [`bench/`](bench), so the
numbers can be reproduced rather than believed. `bench/README.md` records how
they were taken and what the corpus does not cover.

## What it is

```
mvp/ocr_engine/
  device.py       hardware-only Vulkan selection; software rasterisers are refused
  detection.py    DB preprocess -> probability map -> oriented boxes -> unclip
  recognition.py  affine crop -> CTC head -> greedy decode
  engine.py       facade: load once, read(rgb) -> OcrResult
  catalog.py      model sets as data: PP-OCRv6 tiny/small/medium, PP-OCRv5 mobile
```

585 lines of engine, 231 of tests. Dependencies are `ncnn`, `numpy` and
`opencv` — no PaddlePaddle, no ONNX, no polygon clipper.

## Run it

The model graphs are third-party ports and are not vendored here; fetch them
once (see [THIRD-PARTY.md](THIRD-PARTY.md)):

```sh
git clone https://github.com/Avafly/PaddleOCR-ncnn-CPP   # PP-OCRv6, MIT
git clone https://github.com/nihui/ncnn-android-ppocrv5 nihui-port  # PP-OCRv5, BSD-3

python3 -m venv .venv && .venv/bin/pip install ncnn numpy opencv-python-headless pillow
cd mvp
../.venv/bin/python demo.py ../sample-applet.png                 # PP-OCRv6 medium
../.venv/bin/python demo.py ../sample-applet.png --models v6-tiny
../.venv/bin/python -m pytest tests -q                           # 21 tests
```

`demo.py` prints the device it chose, the lines with their coordinates and
confidence, and the GPU's `gpu_busy_percent` while it works — so "it ran on
the GPU" is observable rather than asserted. For the same claim measured per
process, `bench/gpu_fdinfo.py` reads this process's own amdgpu counters.

## Accuracy notes

The port originally followed nihui's reference exactly, including its flat
1.95× box enlargement in place of DB's unclip. That costs real accuracy on
small text — accents and last glyphs shaved off the crop — and
`detection.py` now applies DB's own rule, `area × ratio / perimeter`, which
for a `minAreaRect` is one expression and needs no clipper. It closed the gap
to upstream (CER 0.0223 → 0.0156) and made reading *faster*, because a taller
box yields a narrower 48-px crop.

Known gaps, all recorded in [FINDINGS.md](FINDINGS.md): 90°-rotated text is
unreadable (no orientation classifier), glyph-font icons decode as CJK noise
below ~0.6 confidence, and Hebrew has no pretrained model anywhere in the
Paddle ecosystem.

## Licence

MIT — see [LICENSE](LICENSE). The models and the reference implementation
belong to their authors; [THIRD-PARTY.md](THIRD-PARTY.md) names each one.
