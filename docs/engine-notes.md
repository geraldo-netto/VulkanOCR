# Engine notes — how the port was built and what it costs

Answers "is this truly doable without touching ncnn or PaddleOCR": **yes**,
demonstrated end-to-end on this host's RX 6600 XT.

## What this is

A small engine package with explicit seams for model data, execution policy,
layout, device selection, and measurement:

```
src/vulkanocr/
  device.py       hardware-only Vulkan selection (no-CPU rule; llvmpipe refused)
  detection.py    DB preprocess -> probability map -> oriented boxes -> unclip
  recognition.py  affine crop -> CTC head -> greedy decode
  orientation.py  PP-LCNet 0°/180° text-line correction
  layout.py       conservative inter-region whitespace reconstruction
  engine.py       facade: load once, read(rgb) -> OcrResult
  catalog.py      reusable detector/recognizer specs and named profiles
  options.py      explicit fp32/fp16/int8 ncnn execution policy
  policy.py       opt-in, page-aware filtering of known false readings
  cli.py          the `vulkanocr` command
  parallel.py     detect/orient once, then distribute recognition crops
  scheduling.py   price asymmetric devices and assign crop widths
  workers.py      one recognition-only child process per hardware GPU
  proof.py        per-process DRM proof with an AMD system-wide fallback
tests/            device policy and engine lifecycle (fake runtime),
                  CTC, geometry, orientation, layout, policy, catalogue,
                  multi-GPU scheduling/refusals, scoring, telemetry, live GPU
```

The default model set is **`v6-medium`**. Override per run with
`--models v6-tiny|v6-small|v5-mobile`.

Dependencies: `ncnn`, `numpy`, `opencv`. No PaddlePaddle imports and no
upstream modification. Models come from two third-party ncnn ports of
PaddlePaddle's Apache-2.0 weights: Avafly (MIT) for PP-OCRv6, nihui (BSD-3)
for PP-OCRv5. Avafly also supplies the PP-LCNet orientation graph used by the
v6 profiles. Neither upstream is modified.

By default fp16 packed/storage/arithmetic are all disabled for the measured
fp32 baseline. `InferenceOptions` states fp16 and int8 policy for every net;
`--precision fp32|fp16|int8` exposes the same choice in the CLI and corpus
runner. fp16 reduced the warm, end-to-end corpus p50 from 99 ms to 66 ms
(1.5×) at no measured accuracy cost (`docs/benchmarks.md`); int8 requires a
quantized graph and does not transform the float catalog entries. No quantized
profile is currently catalogued.

## Run

```sh
.venv/bin/vulkanocr samples/sample-applet.png                 # v6 medium
.venv/bin/vulkanocr samples/sample-applet.png --models v6-tiny
.venv/bin/python -m pytest -q
```

## Evidence (2026-08-25, this host)

- The 55-page corpus rerun measures v6-medium at CER 0.0159, WER 0.0390,
  75 % exact, and 99 ms p50; fp16 produces identical text scores at 66 ms p50.
- The 585×770 applet screenshot contains 46 detector crops, assembled into 42
  lines after whitespace reconstruction, at a 665 ms median over three warm
  reads on `AMD Radeon RX 6600 XT (RADV NAVI23)`.
- The CLI attributes DRM fdinfo engine deltas to the selected device when the
  driver exposes them. AMD `gpu_busy_percent` is an explicitly system-wide
  fallback, not process proof.
- Live tests read rendered text at all four cardinal page rotations and assert
  coordinates land on the source; software-only Vulkan is a refusal, not a
  fallback.

## Timing summary

| profile | CER | WER | p50 |
| --- | --- | --- | --- |
| v6-medium fp32 | 0.0159 | 0.0390 | 99 ms |
| v6-medium fp16 | 0.0159 | 0.0390 | 66 ms |
| v6-small fp32 | 0.0181 | 0.0368 | 47 ms |
| v6-tiny fp32 | 0.0163 | 0.0639 | 26 ms |
| v5-mobile fp32 | 0.0351 | 0.1017 | 45 ms |

These are aggregate corpus numbers, not model-only throughput: detection,
orientation, recognition and assembly are included in VulkanOCR rows. See
[`benchmarks.md`](benchmarks.md) for the corpus limits, PaddleOCR/Tesseract
comparison, dense-page timings, and negative batching experiments.

## The multi-GPU pool (`parallel.py`)

`--all-gpus` reads one page with every hardware device. The design facts,
each measured rather than assumed (numbers in `docs/benchmarks.md`):

- One recognition **child process** per device: the ncnn binding holds the GIL
  through `extract`, so a thread pool ran 4x *slower* than one GPU.
- Detection and orientation run once, in-process, on the preferred device;
  workers build
  recognition-only engines (`nets=("rec",)`), so no worker holds a detection
  or orientation net it never runs.
- The dispatcher **prices** every assignment in ms per pixel column — seeded
  by a probe strip at start-up, refined by every finished crop — and grants a
  slow device a cheap tail crop only while its cumulative commitment stays
  under the fast side's projected work. Near-equal devices split the page;
  a sufficiently slow device is left idle when its projected assignment would
  finish later than the fast side's remaining work.
- Every request and reply carries its read's generation, so a read that
  raised cannot leak stale results into the next page; a dead or failing
  worker is retired with a named refusal (`worker-died`/`worker-failed`) and
  the survivors carry the next read, down to the primary-engine fallback.
- On a machine with one hardware device the pool builds no fleet at all: it
  is the primary engine, with the same answers.

Correctness is device-count-independent: lines return in `OcrEngine.read`'s
order, and the live suite asserts the pool's text equals the single engine's.

## Model generations available here

The catalog exposes four profiles: PP-OCRv6 medium (the default), small and
tiny from Avafly, and PP-OCRv5 mobile from nihui. All three v6 profiles attach
Avafly's PP-LCNet x0.25 text-line orientation graph; v5-mobile has no
orientation graph.

Detector and recognizer records independently state paths, blob names,
dictionary/blank convention, and any required precision. A profile pairs the
components and optionally supplies orientation facts. This matters because v6
uses `input`/`output` blobs and dictionaries that include the CTC blank, while
nihui v5 uses `in0`/`out0` and a dictionary without it. A wrong blank
convention does not crash — it silently shifts every character — so these are
catalog data covered by tests rather than engine guesses.

## Known gaps

Current work is tracked in [`TODO.md`](../TODO.md), not the historical spike
report. Its Open table is empty. The Blocked / Deferred table covers Hebrew
model approval and artifacts, validated int8 artifacts, and a checksum-pinned
component model store. Inter-region whitespace reconstruction is implemented;
its deliberate intra-region and bidirectional limits are in
[`whitespace.md`](whitespace.md).
Known glyph-icon readings have an opt-in, page-script-aware filter; see the
README.
