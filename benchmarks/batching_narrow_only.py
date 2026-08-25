"""Historical batching PoC 3: pack narrow, dispatch-bound crops.

Per-call overhead measures ~5 ms, so a 43-px crop costs 6.5 ms of which about
1.4 ms is arithmetic. Wide crops are compute-bound and gain nothing from
packing — and packing them corrupts the decode, because the encoder mixes
across the strip. This packs only crops below a width threshold and keeps the
rest one to a call, then checks both halves of that claim.

Measured on 2026-08-21. The PoC predates ``OcrEngine.crops()`` returning
``(region, patch)`` pairs and needs an unpacking adapter before rerun.
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from benchmark_harness import benchmark_engine, require_text_crops
from scoring import load_rgb
from strips import decode_spans, pack


def read_one(engine, crop):
    return engine.decode(engine.logits(crop))[0]


def read_packed(engine, crops):
    strip, spans = pack(crops)
    return decode_spans(engine, strip, spans)


def main() -> int:
    image = sys.argv[1]
    rgb = load_rgb(image)
    with benchmark_engine("v6-medium") as engine:
        return _measure(engine, rgb)


def _measure(engine, rgb) -> int:
    crops = require_text_crops(engine, rgb)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
