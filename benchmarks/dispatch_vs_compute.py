"""Is recognition dispatch-bound or compute-bound? And what does fp16 buy?"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from typing import Any

import ncnn

# The ncnn wheel ships no stubs, so the module is a seam like the engine's.
ncnn_runtime: Any = ncnn
import numpy as np
from scoring import load_rgb

from vulkanocr.catalog import models_for
from vulkanocr.engine import OcrEngine

image = sys.argv[1]
rgb = load_rgb(image)


def build(model_set, fp16):
    return OcrEngine(models_for(model_set), use_fp16=fp16)


def recognise_all(engine, crops):
    return [engine.recognise(c) for c in crops]


for model_set in ("v6-medium", "v6-tiny"):
    for fp16 in (False, True):
        engine = build(model_set, fp16)
        crops = [patch for _region, patch in engine.crops(rgb)]
        pixels = sum(c.shape[0] * c.shape[1] for c in crops)
        recognise_all(engine, crops[:3])
        start = time.perf_counter()
        for _ in range(3):
            recognise_all(engine, crops)
        ms = (time.perf_counter() - start) / 3 * 1000
        engine.close()
        print(
            f"{model_set:10} fp16={str(fp16):5} rec {ms:7.1f} ms for {len(crops)} crops "
            f"({pixels / 1e6:.2f} MPix) = {ms / len(crops):5.2f} ms/crop, "
            f"{ms / (pixels / 1e6):6.1f} ms/MPix"
        )

# Dispatch overhead measured directly: the same crop, N times, vs one crop N times wider.
engine = build("v6-medium", False)
crops = [patch for _region, patch in engine.crops(rgb)]
narrow = min(crops, key=lambda c: c.shape[1])
engine.recognise(narrow)
start = time.perf_counter()
for _ in range(20):
    engine.recognise(narrow)
per_call = (time.perf_counter() - start) / 20 * 1000
wide = np.tile(narrow, (1, 20, 1))
engine.recognise(wide)
start = time.perf_counter()
for _ in range(3):
    engine.recognise(wide)
wide_ms = (time.perf_counter() - start) / 3 * 1000
print(f"\nsmallest crop ({narrow.shape[1]}px wide): {per_call:.1f} ms each")
print(
    f"the same content 20x wider ({wide.shape[1]}px): {wide_ms:.1f} ms once "
    f"vs {per_call * 20:.1f} ms as 20 calls -> "
    f"per-call overhead ~{(per_call * 20 - wide_ms) / 20:.2f} ms"
)
