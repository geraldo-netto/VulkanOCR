"""Batching PoC: recognise the crops of one page concurrently.

ncnn has no batch dimension a convolution graph can use — `Mat` is (w, h, c)
and the exported PP-OCR rec graph takes one 48-high strip at a time — so
"batching" here means what upstream's C++ does with OpenMP: several
extractors in flight at once over the same loaded net, so the GPU has work
queued while the CPU packs the next crop.

Measured against the sequential path on the same image, same models, same
device, with the decoded text compared line for line.
"""

from __future__ import annotations

import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from scoring import load_rgb

from vulkanocr.catalog import models_for
from vulkanocr.engine import OcrEngine


def sequential(engine, patches):
    return [engine.recognise(patch) for _region, patch in patches]


def concurrent(engine, patches, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda pair: engine.recognise(pair[1]), patches))


def timed(call, runs=3):
    call()
    start = time.perf_counter()
    for _ in range(runs):
        out = call()
    return (time.perf_counter() - start) / runs * 1000, out


def main() -> int:
    image, model_set = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "v6-medium")
    rgb = load_rgb(image)
    engine = OcrEngine(models_for(model_set))
    patches = engine.crops(rgb)
    print(f"{model_set} on {engine.device_name}: {len(patches)} crops")

    base_ms, base_out = timed(lambda: sequential(engine, patches))
    print(f"  sequential          {base_ms:8.1f} ms   ({base_ms / len(patches):5.1f} ms/crop)")

    for workers in (2, 4, 8, 16):
        try:
            ms, out = timed(lambda w=workers: concurrent(engine, patches, w))
        except Exception as error:  # noqa: BLE001 - a PoC reports what it hits
            print(f"  {workers:2} extractors      failed: {type(error).__name__}: {error}")
            continue
        same = [text for text, _c in out] == [text for text, _c in base_out]
        print(
            f"  {workers:2} extractors     {ms:8.1f} ms   {base_ms / ms:4.2f}x   "
            f"text identical: {same}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
