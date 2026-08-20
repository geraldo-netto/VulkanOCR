"""Batching PoC 2: one net call over several crops packed side by side.

The rec net takes a 48-high strip of any width and emits one CTC timestep per
few input columns, so k crops separated by a white gap wider than the
network's receptive field *may* decode independently in a single pass. Whether
they do is a property of the graph — a purely convolutional encoder keeps
them independent, an attention encoder mixes them — so this measures it
rather than assuming it: every batched line is compared with the same crop
read on its own.
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scoring import load_rgb

from vulkanocr.catalog import models_for
from vulkanocr.detection import detect_regions
from vulkanocr.engine import OcrEngine
from vulkanocr.recognition import MEAN, NORM, crop_region, decode_ctc

GAP = 32  # white columns between crops


def crops_of(engine, rgb):
    regions = detect_regions(
        engine._runtime, engine._det, rgb, engine._target_size, engine._models.blobs
    )
    return [crop for crop in (crop_region(rgb, region) for region in regions) if crop.size]


def logits_for(engine, strip):
    height, width = strip.shape[:2]
    mat = engine._runtime.Mat.from_pixels(
        np.ascontiguousarray(strip), engine._runtime.Mat.PixelType.PIXEL_RGB2BGR, width, height
    )
    mat.substract_mean_normalize(MEAN, NORM)
    extractor = engine._rec.create_extractor()
    try:
        extractor.input(engine._models.blobs[0], mat)
        code, out = extractor.extract(engine._models.blobs[1])
        if code != 0:
            raise RuntimeError("extraction failed")
        logits = np.array(out)
    finally:
        del extractor
    return logits[0] if logits.ndim == 3 else logits


def pack(crops):
    width = sum(crop.shape[1] for crop in crops) + GAP * (len(crops) - 1)
    strip = np.full((48, width, 3), 255, dtype=np.uint8)
    spans, x = [], 0
    for crop in crops:
        strip[:, x : x + crop.shape[1]] = crop
        spans.append((x, x + crop.shape[1]))
        x += crop.shape[1] + GAP
    return strip, spans


def batched_texts(engine, crops, group):
    offset = engine._models.ctc_offset
    texts = []
    for start in range(0, len(crops), group):
        chunk = crops[start : start + group]
        strip, spans = pack(chunk)
        logits = logits_for(engine, strip)
        steps = logits.shape[0]
        scale = steps / strip.shape[1]
        for left, right in spans:
            lo, hi = (
                int(round(left * scale)),
                max(int(round(right * scale)), int(round(left * scale)) + 1),
            )
            texts.append(decode_ctc(logits[lo:hi], engine._characters, offset)[0])
    return texts


def single_texts(engine, crops):
    offset = engine._models.ctc_offset
    return [decode_ctc(logits_for(engine, crop), engine._characters, offset)[0] for crop in crops]


image, model_set = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "v6-medium")
rgb = load_rgb(image)
engine = OcrEngine(models_for(model_set))
crops = crops_of(engine, rgb)
widths = [crop.shape[1] for crop in crops]
print(f"{model_set}: {len(crops)} crops, widths {min(widths)}..{max(widths)}")

single_texts(engine, crops[:2])
start = time.perf_counter()
for _ in range(3):
    base = single_texts(engine, crops)
base_ms = (time.perf_counter() - start) / 3 * 1000
print(f"  one call per crop     {base_ms:8.1f} ms")

for group in (2, 4, 8, 16, len(crops)):
    batched_texts(engine, crops[: min(group, len(crops))], group)
    start = time.perf_counter()
    for _ in range(3):
        got = batched_texts(engine, crops, group)
    ms = (time.perf_counter() - start) / 3 * 1000
    same = sum(1 for a, b in zip(base, got, strict=True) if a == b)
    print(
        f"  {group:3} crops per call   {ms:8.1f} ms   {base_ms / ms:4.2f}x   "
        f"identical {same}/{len(base)}"
    )
