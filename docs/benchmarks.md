# Benchmark and batching PoC (2026-08-20)

Two questions, both answered by measurement on this host: how the ncnn/Vulkan
port compares with the engines it would replace, and whether batching the
recognition pass is worth building.

Everything here runs against the installed `vulkanocr` package. Nothing is installed into
OmniTensor and nothing in `../mvp` was changed to make these run.

## The corpus

`benchmarks/make_corpus.py` renders 5 texts × 11 degradations = 55 images with exact
ground truth: three sizes, three fonts, 5° and 12° skew, blur, noise, JPEG
q30, and a faded scan. Synthetic on purpose — a comparison needs identical
inputs and exact truth for every engine, and a hand-transcribed scan holdout
is days of work. It is the trade this makes, and the acceptance corpus
OMNI-0509 asks for is still the honest one.

`benchmarks/scoring.py` scores a read: CER and WER as total edit distance over total
length (never a mean of per-image rates), whitespace-normalised, reading order
ignored — a caller consumes the text, not the box order.

```sh
.venv/bin/python benchmarks/make_corpus.py /tmp/ocrcorpus
.venv/bin/python benchmarks/read_with_vulkanocr.py /tmp/ocrcorpus v6-medium
python3 benchmarks/read_with_tesseract.py /tmp/ocrcorpus 6
<paddle venv>/bin/python benchmarks/read_with_paddleocr.py /tmp/ocrcorpus
python3 benchmarks/compare_engines.py /tmp
```

## Accuracy and speed

| engine | device | CER | WER | exact | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| PaddleOCR 3.7.0 (PP-OCRv5 mobile) | CPU | 0.0154 | 0.0498 | 73% | 1345 ms | 1638 ms |
| spike v6-medium | RX 6600 XT | 0.0223 | 0.0942 | 51% | 114 ms | 142 ms |
| spike v6-medium, fp16 | RX 6600 XT | 0.0225 | 0.0952 | 51% | 82 ms | 113 ms |
| spike v6-tiny | RX 6600 XT | 0.0256 | 0.1429 | 24% | 24 ms | 32 ms |
| Tesseract 5.3.4 psm6 | CPU | 0.0459 | 0.1255 | 58% | 98 ms | 111 ms |
| spike v5-mobile | RX 6600 XT | 0.0642 | 0.2922 | 22% | 50 ms | 76 ms |

Dense UI page (`samples/sample-applet.png`, 585×770): Tesseract 264 ms wall / 827 ms
CPU; spike v6-medium 727 ms wall / 1443 ms CPU; PaddleOCR 8995 ms wall /
**89270 ms CPU**. PaddleOCR on this desk is CPU-only —
`is_compiled_with_cuda()` and `is_compiled_with_rocm()` are both false and
Paddle has no Vulkan backend — so the port is not a faster PaddleOCR, it is
the only way to put OCR on this GPU at all.

## That it really is Vulkan

`gpu_proof.py` reads this process's own amdgpu counters: **+1314 ms of
`drm-engine-compute` per 750 ms read**, 721 MiB of VRAM and 106 MiB of GTT
held. `vulkan_vs_cpu.py` runs the same models with `use_vulkan_compute` off:
759 ms on Vulkan against 1211 ms on ncnn's CPU lane, identical output.
`phase_timings.py` shows 748 of 752 ms inside the nets — 39 ms detection (one
call) and 708 ms recognition (46 calls).

## The batching PoC: three attempts, all negative

| attempt | script | result |
| --- | --- | --- |
| concurrent extractors, 2–16 threads | `batching_threads.py` | 0.96–0.99× — no gain. The Python binding serialises; text identical, so it is safe, just pointless |
| pack every crop into one wide strip | `batching_one_strip.py` | 1.01–1.05×, and **only 37/46 lines identical** — the encoder mixes across the strip, so the decode is wrong |
| pack only crops narrower than N | `batching_narrow_only.py` | 664–795 ms against a 734 ms baseline, inside the noise, still 44/46 identical |

`dispatch_vs_compute.py` explains it. Per-call overhead is ~5.1 ms: the smallest
crop (43 px) costs 6.5 ms, while the same content 20× wider costs 28.9 ms in
one call against 130.3 ms as twenty. So dispatch overhead is real — but this
page's crops run to 1088 px, and at 902 ms/MPix the arithmetic dominates the
total. Packing then adds gap columns and loses lines to cross-talk, which is
why the measured gain is nil.

## What does pay

| lever | measured |
| --- | --- |
| fp16 on v6-medium | recognition 735 → 492 ms (**1.49×**); corpus p50 114 → 82 ms; CER 0.0223 → 0.0225 — **unchanged inside noise** |
| v6-tiny instead of v6-medium | recognition 735 → 151 ms (**4.9×**) at CER 0.0223 → 0.0256 |

fp16 is disabled here to match the service's Vulkan policy. On this workload
that policy costs a third of the wall clock and buys no accuracy, which is a
decision worth taking deliberately rather than by inheritance.
