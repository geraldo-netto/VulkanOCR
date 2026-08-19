# PaddleOCR on ncnn over Vulkan — OMNI-0351 spike

A self-contained feasibility spike: OCR running on this host's GPU through
ncnn's Vulkan backend, using PaddleOCR models, **without modifying ncnn or
PaddleOCR and without installing anything into OmniTensor**.

Nothing here is installed, imported, or reachable by the running service. The
folder is a sealed experiment: delete it and nothing else changes.

- `FINDINGS.md` — what the first pass proved, and what it cost
- `mvp/README.md` — the engine package, its evidence, and the model comparison

## Requirements

Already satisfied on this machine; listed so the folder stands alone.

| Need | Why | Check |
| --- | --- | --- |
| A hardware Vulkan GPU | inference lane; software rasterisers are refused, not used as fallback | `vulkaninfo --summary` |
| Mesa/RADV or vendor driver | ncnn talks Vulkan, not CUDA/ROCm | `ls /dev/dri/renderD*` |
| Python 3.12 | matches the bundled virtualenv | `python3 -V` |

No system packages are needed. No `pip install -e`, no `setup.py`, no
`PYTHONPATH` edits in your shell profile.

## Run it

Everything runs out of the bundled virtualenv at `.venv/`. The engine package
is imported from the working directory, not from site-packages.

```sh
cd /backups/disk2/projects/cinnamon/ocr-ncnn-spike/mvp

# read an image with the default model set (PP-OCRv6 medium)
../.venv/bin/python demo.py ../sample-applet.png

# a different tier: fastest, balanced, or the previous generation
../.venv/bin/python demo.py ../sample-applet.png --models v6-tiny
../.venv/bin/python demo.py ../sample-applet.png --models v6-small
../.venv/bin/python demo.py ../sample-applet.png --models v5-mobile

# your own file
../.venv/bin/python demo.py ~/Pictures/receipt.jpg

# the test suite: 21 tests, device policy and CTC decode need no GPU
../.venv/bin/python -m pytest tests/ -q
```

`demo.py` prints the selected device, the recognised lines with their
coordinates and confidence, warm-pass timings, and the GPU's
`gpu_busy_percent` sampled from sysfs while it works — so "it ran on the GPU"
is observable rather than asserted.

The first throwaway script from the original spike is still at
`spike_ocr.py` in this folder. It predates the engine package and duplicates
its logic in one file; keep it only as a record of the first pass.

```sh
.venv/bin/python spike_ocr.py sample-synthetic.png --models mobile
```

## Use the engine from your own script

No installation: put `mvp/` on the path for the one command.

```sh
PYTHONPATH=/backups/disk2/projects/cinnamon/ocr-ncnn-spike/mvp \
  /backups/disk2/projects/cinnamon/ocr-ncnn-spike/.venv/bin/python - <<'PY'
import cv2
from ocr_engine import OcrEngine, models_for

engine = OcrEngine(models_for())            # PP-OCRv6 medium by default
rgb = cv2.cvtColor(cv2.imread("page.png"), cv2.COLOR_BGR2RGB)
result = engine.read(rgb)

print(result.device_name)
for line in result.lines:
    print(f"{line.confidence:.2f}  {line.text}")
PY
```

`models_for("v6-tiny")` and friends select another set; `CATALOG` lists them
with a one-line note each.

## What is in the folder

| Path | Size | What |
| --- | --- | --- |
| `mvp/` | 184 K | the engine package, demo, and tests — the part worth keeping |
| `.venv/` | ~400 M | ncnn 1.0.20260526, numpy, OpenCV, pytest |
| `nihui-port/` | 396 M | PP-OCRv5 ncnn models + reference C++ (BSD-3, Tencent) |
| `PaddleOCR-ncnn-CPP/` | 281 M | PP-OCRv3/4/5/6 models, orientation classifiers, reference C++ (MIT, Avafly) |
| `avafly-v0.3.0.tar.gz` | 231 M | the pinned download the models came from; kept for provenance |
| `sample-*.png` | small | synthetic text, an applet screenshot, rotations, noise |

Models are third-party ncnn conversions of PaddlePaddle's Apache-2.0 PP-OCR
weights. Neither ncnn nor PaddleOCR is modified, forked, or vendored, and
PaddlePaddle itself is never installed — it is a source of weights, offline,
once.

## Rebuilding the virtualenv

Only needed if `.venv/` is deleted or the folder moves again.

```sh
cd /backups/disk2/projects/cinnamon/ocr-ncnn-spike
rm -rf .venv
python3 -m venv .venv
.venv/bin/pip install "ncnn>=1.0.20260526" numpy opencv-python-headless pillow pytest
```

## Troubleshooting

**`bad interpreter` from `.venv/bin/pip` or `pytest`** — the folder moved and
the console scripts carry absolute shebangs. Either rebuild the virtualenv as
above, or call modules directly, which is path-independent:
`.venv/bin/python -m pytest`, `.venv/bin/python -m pip`.

**`no hardware Vulkan device is present`** — deliberate. The engine refuses
`llvmpipe`, the software rasteriser, rather than pretending a CPU run is a GPU
run. Check `vulkaninfo --summary` for a real device.

**Screens of `queueC=…  fp16-p/s/u/a=…` before any output** — that is ncnn
enumerating Vulkan devices on stderr, not an error. Filter with
`2>&1 | grep -v "queueC=\|fp16-\|subgroup="`.

**Recognised text is fluent nonsense** — the dictionary convention is wrong for
that model. Sets whose keys file carries the CTC blank on its first line need
`dictionary_includes_blank=True`; the catalogue records this per model, and
getting it wrong shifts every character by one without any error.

**Icons decoded as CJK characters** — glyph-font icons look like text to a
detector. They land below roughly 0.6 confidence; filter on `line.confidence`.

## Status

Feasibility is proven and the folder is a dead end by design: it is not wired
into OmniTensor, not packaged, and not on any install path. Turning it into a
real workload means the provider distribution, a digest-pinned model recipe,
the `AdapterKind.OCR` wiring, and a licensed acceptance corpus — see
`FINDINGS.md` for that shape and its remaining costs.
