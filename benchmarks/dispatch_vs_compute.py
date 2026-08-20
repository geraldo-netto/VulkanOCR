"""Is recognition dispatch-bound or compute-bound? And what does fp16 buy?"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import cv2
import ncnn
import numpy as np

from vulkanocr.catalog import models_for
from vulkanocr.detection import detect_regions
from vulkanocr.engine import OcrEngine
from vulkanocr.recognition import crop_region, recognise_patch

image = sys.argv[1]
rgb = np.ascontiguousarray(cv2.imread(image)[:, :, ::-1])


def build(model_set, fp16):
    class Tuned(OcrEngine):
        def _load_net(self, param_path):
            net = ncnn.Net()
            net.opt.use_vulkan_compute = True
            net.opt.use_fp16_packed = fp16
            net.opt.use_fp16_storage = fp16
            net.opt.use_fp16_arithmetic = fp16
            net.load_param(str(param_path))
            net.load_model(str(param_path.with_suffix(".bin")))
            net.opt.use_vulkan_compute = True
            return net

    engine = Tuned(models_for(model_set))
    engine._device_index = None
    return engine


def crops_of(engine):
    regions = detect_regions(
        engine._runtime, engine._det, rgb, engine._target_size, engine._models.blobs
    )
    return [c for c in (crop_region(rgb, r) for r in regions) if c.size]


def recognise_all(engine, crops):
    offset = 0 if engine._models.dictionary_includes_blank else 1
    return [
        recognise_patch(
            engine._runtime, engine._rec, c, engine._characters, engine._models.blobs, offset
        )
        for c in crops
    ]


for model_set in ("v6-medium", "v6-tiny"):
    for fp16 in (False, True):
        engine = build(model_set, fp16)
        crops = crops_of(engine)
        pixels = sum(c.shape[0] * c.shape[1] for c in crops)
        recognise_all(engine, crops[:3])
        start = time.perf_counter()
        for _ in range(3):
            recognise_all(engine, crops)
        ms = (time.perf_counter() - start) / 3 * 1000
        print(
            f"{model_set:10} fp16={str(fp16):5} rec {ms:7.1f} ms for {len(crops)} crops "
            f"({pixels / 1e6:.2f} MPix) = {ms / len(crops):5.2f} ms/crop, "
            f"{ms / (pixels / 1e6):6.1f} ms/MPix"
        )

# Dispatch overhead measured directly: the same crop, N times, vs one crop N times wider.
engine = build("v6-medium", False)
crops = crops_of(engine)
narrow = min(crops, key=lambda c: c.shape[1])
offset = 0 if engine._models.dictionary_includes_blank else 1
recognise_patch(
    engine._runtime, engine._rec, narrow, engine._characters, engine._models.blobs, offset
)
start = time.perf_counter()
for _ in range(20):
    recognise_patch(
        engine._runtime, engine._rec, narrow, engine._characters, engine._models.blobs, offset
    )
per_call = (time.perf_counter() - start) / 20 * 1000
wide = np.tile(narrow, (1, 20, 1))
recognise_patch(
    engine._runtime, engine._rec, wide, engine._characters, engine._models.blobs, offset
)
start = time.perf_counter()
for _ in range(3):
    recognise_patch(
        engine._runtime, engine._rec, wide, engine._characters, engine._models.blobs, offset
    )
wide_ms = (time.perf_counter() - start) / 3 * 1000
print(f"\nsmallest crop ({narrow.shape[1]}px wide): {per_call:.1f} ms each")
print(
    f"the same content 20x wider ({wide.shape[1]}px): {wide_ms:.1f} ms once "
    f"vs {per_call * 20:.1f} ms as 20 calls -> "
    f"per-call overhead ~{(per_call * 20 - wide_ms) / 20:.2f} ms"
)
