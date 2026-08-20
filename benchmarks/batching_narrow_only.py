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
from scoring import load_rgb
from strips import decode_spans, pack

from vulkanocr.catalog import models_for
from vulkanocr.engine import OcrEngine


def logits_for(engine, strip):
    """The engine's own preprocessing and extraction, on one packed strip."""
    return engine.logits(strip)


def read_one(engine, crop):
    return engine.decode(logits_for(engine, crop))[0]


def read_packed(engine, crops):
    strip, spans = pack(crops)
    return decode_spans(engine, strip, spans)


image = sys.argv[1]
rgb = load_rgb(image)
engine = OcrEngine(models_for("v6-medium"))
crops = [patch for _region, patch in engine.crops(rgb)]
base = [read_one(engine, c) for c in crops]

for threshold in (0, 96, 160, 256, 10_000):
    narrow = [i for i, c in enumerate(crops) if c.shape[1] <= threshold]
    wide = [i for i, c in enumerate(crops) if c.shape[1] > threshold]

    def run(narrow=narrow, wide=wide):
        got = dict.fromkeys(range(len(crops)))
        for index in wide:
            got[index] = read_one(engine, crops[index])
        if narrow:
            for text, index in zip(
                read_packed(engine, [crops[i] for i in narrow]), narrow, strict=True
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
