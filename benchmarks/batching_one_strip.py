"""Historical batching PoC 2: pack several crops into one net call.

The rec net takes a 48-high strip of any width and emits one CTC timestep per
few input columns, so k crops separated by a white gap wider than the
network's receptive field *may* decode independently in a single pass. Whether
they do is a property of the graph — a purely convolutional encoder keeps
them independent, an attention encoder mixes them — so this measures it
rather than assuming it: every batched line is compared with the same crop
read on its own.

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


def batched_texts(engine, crops, group):
    texts = []
    for start in range(0, len(crops), group):
        strip, spans = pack(crops[start : start + group])
        texts.extend(decode_spans(engine, strip, spans))
    return texts


def single_texts(engine, crops):
    return [engine.decode(engine.logits(crop))[0] for crop in crops]


def main() -> int:
    image, model_set = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "v6-medium")
    rgb = load_rgb(image)
    with benchmark_engine(model_set) as engine:
        return _measure(engine, rgb, model_set)


def _measure(engine, rgb, model_set) -> int:
    crops = require_text_crops(engine, rgb)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
