"""Batching PoC 3: pack only the narrow crops, where dispatch dominates.

Per-call overhead measures ~5 ms, so a 43-px crop costs 6.5 ms of which about
1.4 ms is arithmetic. Wide crops are compute-bound and gain nothing from
packing — and packing them corrupts the decode, because the encoder mixes
across the strip. This packs only crops below a width threshold and keeps the
rest one to a call, then checks both halves of that claim.
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

GAP = 32


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


def read_one(engine, crop, offset):
    return decode_ctc(logits_for(engine, crop), engine._characters, offset)[0]


def read_packed(engine, crops, offset):
    width = sum(c.shape[1] for c in crops) + GAP * (len(crops) - 1)
    strip = np.full((48, width, 3), 255, dtype=np.uint8)
    spans, x = [], 0
    for crop in crops:
        strip[:, x : x + crop.shape[1]] = crop
        spans.append((x, x + crop.shape[1]))
        x += crop.shape[1] + GAP
    logits = logits_for(engine, strip)
    scale = logits.shape[0] / width
    out = []
    for left, right in spans:
        lo = int(round(left * scale))
        hi = max(int(round(right * scale)), lo + 1)
        out.append(decode_ctc(logits[lo:hi], engine._characters, offset)[0])
    return out


image = sys.argv[1]
rgb = load_rgb(image)
engine = OcrEngine(models_for("v6-medium"))
offset = engine._models.ctc_offset
regions = detect_regions(
    engine._runtime, engine._det, rgb, engine._target_size, engine._models.blobs
)
crops = [c for c in (crop_region(rgb, r) for r in regions) if c.size]
base = [read_one(engine, c, offset) for c in crops]

for threshold in (0, 96, 160, 256, 10_000):
    narrow = [i for i, c in enumerate(crops) if c.shape[1] <= threshold]
    wide = [i for i, c in enumerate(crops) if c.shape[1] > threshold]

    def run(narrow=narrow, wide=wide):
        got = dict.fromkeys(range(len(crops)))
        for index in wide:
            got[index] = read_one(engine, crops[index], offset)
        if narrow:
            for text, index in zip(
                read_packed(engine, [crops[i] for i in narrow], offset), narrow, strict=True
            ):
                got[index] = text
        return [got[i] for i in range(len(crops))]

    run()
    start = time.perf_counter()
    for _ in range(3):
        got = run()
    ms = (time.perf_counter() - start) / 3 * 1000
    same = sum(1 for a, b in zip(base, got, strict=True) if a == b)
    print(
        f"  pack crops <= {threshold:5} px ({len(narrow):2} packed, {len(wide):2} single): "
        f"{ms:7.1f} ms   identical {same}/{len(base)}"
    )
