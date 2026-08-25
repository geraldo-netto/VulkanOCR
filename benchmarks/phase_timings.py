"""Where the wall clock goes: Vulkan nets vs the Python/OpenCV glue.

Rewritten off the monkeypatch: it used to replace three functions inside
`vulkanocr.engine`'s module namespace, which broke the moment the engine
composed them differently. The public detection and recognition seams are
timed instead. Optional text-line orientation and result assembly are excluded,
so this profiles the expensive phases in isolation rather than reconstructing
the full `read()` wall clock.
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from benchmark_harness import benchmark_engine, require_nonempty, require_text_crops
from scoring import load_rgb

from vulkanocr.recognition import crop_region

RUNS = 5


def main() -> int:
    rgb = load_rgb(sys.argv[1])
    model_set = sys.argv[2] if len(sys.argv) > 2 else "v6-medium"
    with benchmark_engine(model_set) as engine:
        return _measure(engine, rgb)


def _measure(engine, rgb) -> int:
    require_text_crops(engine, rgb)
    engine.read(rgb)  # warm pass pays for shader compilation

    detect_ms = crop_ms = rec_ms = 0.0
    crop_count = lines = 0
    start = time.perf_counter()
    for _ in range(RUNS):
        tick = time.perf_counter()
        regions = engine.detect(rgb)
        detect_ms += time.perf_counter() - tick

        tick = time.perf_counter()
        patches = [patch for patch in (crop_region(rgb, r) for r in regions) if patch.size]
        require_nonempty(patches, "text crops detected")
        crop_ms += time.perf_counter() - tick

        tick = time.perf_counter()
        read = [engine.recognise(patch) for patch in patches]
        rec_ms += time.perf_counter() - tick
        crop_count += len(patches)
        lines = sum(1 for text, _confidence in read if text)
    wall = time.perf_counter() - start

    print(f"lines {lines}, crops/read {crop_count // RUNS}")
    print(f"  detection (1 net call)   {detect_ms / RUNS * 1000:7.1f} ms")
    print(f"  cropping  (opencv, cpu)  {crop_ms / RUNS * 1000:7.1f} ms")
    print(f"  recognition ({crop_count // RUNS} net calls) {rec_ms / RUNS * 1000:7.1f} ms")
    print(
        f"  everything else          {(wall - detect_ms - crop_ms - rec_ms) / RUNS * 1000:7.1f} ms"
    )
    print(f"  wall per read            {wall / RUNS * 1000:7.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
