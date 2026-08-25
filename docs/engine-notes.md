# Engine notes — how the port was built and what it costs

Answers "is this truly doable without touching ncnn or PaddleOCR": **yes**,
demonstrated end-to-end on this host's RX 6600 XT.

## What this is

A minimal but clean engine package with the exact shape the eventual
`omnitensor-ocr-engine` distribution would have:

```
src/vulkanocr/
  device.py       hardware-only Vulkan selection (no-CPU rule; llvmpipe refused)
  detection.py    DB preprocess -> probability map -> oriented boxes -> unclip
  recognition.py  affine crop -> CTC head -> greedy decode
  engine.py       facade: load once, read(rgb) -> OcrResult
  catalog.py      known model sets as data; default is PP-OCRv6 medium
  cli.py          the `vulkanocr` command
  parallel.py     one recognition process per GPU, work priced per assignment
  proof.py        sysfs gpu_busy_percent sampling, as a context manager
tests/            device policy and engine lifecycle (fake runtime),
                  CTC vectors, detection geometry, catalogue facts, the
                  assembler, the pool's dispatcher and refusals (GPU-free),
                  the scorer and measurement loop, the busy sampler, live GPU
```

The default model set is **`v6-medium`**. Override per run with
`--models v6-tiny|v6-small|v5-mobile`.

Dependencies: `ncnn`, `numpy`, `opencv`. No paddle, no omnitensor imports,
no upstream modification. Models come from two third-party ncnn ports of
PaddlePaddle's Apache-2.0 weights: Avafly (MIT) for PP-OCRv6, nihui (BSD-3)
for PP-OCRv5. Neither upstream is modified.

By default fp16 packed/storage/arithmetic are all disabled, matching the
service's Vulkan policy. `InferenceOptions` states fp16 and int8 policy for
every net; `--precision fp32|fp16|int8` exposes the same choice in the CLI and
corpus runner. fp16 bought 1.5x on recognition at no measured accuracy cost
(`docs/benchmarks.md`); int8 requires a quantized graph and does not transform
the float catalog entries.

## Run

```sh
.venv/bin/vulkanocr samples/sample-applet.png                 # v6 medium
.venv/bin/vulkanocr samples/sample-applet.png --models v6-tiny
.venv/bin/python -m pytest -q
```

## Evidence (2026-08-15, this host)

- Applet screenshot (585×770, dense 12 px UI text) on the v6-medium default:
  46 lines, all known strings present, no low-confidence noise;
  `device: AMD Radeon RX 6600 XT (RADV NAVI23)`, ~0.9 s warm.
- While reading, `card1` (PCI 1002:73FF = RX 6600 XT) `gpu_busy_percent`
  mean ~91 %, max 99 %, against a ~39 % ambient baseline (the omnitensor
  service probes the GPU twice a second, so ambient is not zero).
- Live tests render text with OpenCV, read it back exactly, and assert
  coordinates land on the text; software-only Vulkan is a refusal, not a
  fallback.

## Timing matrix (warm, ms/read, same image)

| models | vulkan | cpu reference |
| --- | --- | --- |
| mobile | 618 | 514 |
| server | 1031 | 1794 |

Honest finding: with **mobile** models and 57 tiny sequential crops, the CPU
reference is faster — per-crop Vulkan dispatch overhead dominates nets this
small. **Server** models are 1.7× faster on the GPU. Consequences for the
real integration:

- The no-CPU rule is about keeping the desktop's CPU free and the lane
  qualified, not about winning microbenchmarks; mobile-on-Vulkan at ~0.6 s
  per screenshot is well inside interactive budget.
- If throughput matters, the knobs are: server models (GPU-favoured),
  batching crops per extractor, and reusing one extractor per page. Not a
  blocker for the provider work.

## The multi-GPU pool (`parallel.py`)

`--all-gpus` reads one page with every hardware device. The design facts,
each measured rather than assumed (numbers in `docs/benchmarks.md`):

- One **process** per device: the ncnn binding holds the GIL through
  `extract`, so a thread pool ran 4x *slower* than one GPU.
- Detection runs once, in-process, on the preferred device; workers build
  recognition-only engines (`nets=("rec",)`), so no worker holds a detection
  net it never runs.
- The dispatcher **prices** every assignment in ms per pixel column — seeded
  by a probe strip at start-up, refined by every finished crop — and grants a
  slow device a cheap tail crop only while its cumulative commitment stays
  under the fast side's projected work. Near-equal devices split the page;
  a 7.5x-slower card contributes a little and cannot hurt.
- Every request and reply carries its read's generation, so a read that
  raised cannot leak stale results into the next page; a dead or failing
  worker is retired with a named refusal (`worker-died`/`worker-failed`) and
  the survivors carry the next read, down to the primary-engine fallback.
- On a machine with one hardware device the pool builds no fleet at all: it
  is the primary engine, with the same answers.

Correctness is device-count-independent: lines return in `OcrEngine.read`'s
order, and the live suite asserts the pool's text equals the single engine's.

## Model generations available here

`nihui-port/` carries PP-OCR**v5** (nihui, ncnn's author, BSD-3, 2026-05-27).
`PaddleOCR-ncnn-CPP/models/` carries PP-OCR**v3/v4/v5/v6** plus two textline
orientation classifiers (Avafly, MIT, release v0.3.0 2026-06-13). PP-OCRv6 is
the current generation in PaddleOCR main; the repository's latest release is
v3.7.0 (2026-06-11).

Driving v6 needed no engine change — only two data facts per model record:
the blob names (`input`/`output` instead of `in0`/`out0`) and whether the
dictionary carries the CTC blank as its first line (v6 keys files do, the
nihui v5 dictionary does not). Both are fields on `OcrModels`, both covered by
tests. Same image, same GPU, corrected offsets:

| models | lines | ms/read | known strings | low-confidence lines |
| --- | --- | --- | --- | --- |
| v5 mobile (nihui) | 56 | 531 | 7/7 | 3 |
| v6 tiny | 51 | 372 | 7/7 | 1 |
| v6 small | 51 | 736 | 7/7 | 2 |
| v6 medium | 46 | 1000 | 7/7 | 0 |

v6 tiny is both faster and cleaner than v5 mobile; v6 medium emits no
low-confidence junk at all. A wrong blank convention does not crash — it
shifts every character silently, which is why it is a stated field and a test
rather than a guess.

## Known gaps

Current work is tracked in [`TODO.md`](../TODO.md), not the historical spike
report. It includes glyph-font icons decoded as CJK noise, occasional dropped
spaces between words, and the Hebrew model/provenance decision.
