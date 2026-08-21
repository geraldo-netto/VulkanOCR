"""Same models, same image: Vulkan lane against ncnn's own CPU lane."""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

# The ncnn wheel ships no stubs, so the module is a seam like the engine's.


def main() -> int:
    from scoring import load_rgb

    from vulkanocr.catalog import models_for
    from vulkanocr.engine import OcrEngine

    rgb = load_rgb(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else "v6-medium"

    with OcrEngine(models_for(name)) as engine:
        engine.read(rgb)
        start = time.perf_counter()
        for _ in range(3):
            vulkan_result = engine.read(rgb)
        vulkan = (time.perf_counter() - start) / 3

    # The same engine with the Vulkan knob off: still the one checked,
    # device-pinned loader, so a corrupt model refuses instead of measuring
    # an empty graph. Built after the Vulkan engine is closed, so the two
    # never hold the card's memory at once.
    with OcrEngine(models_for(name), use_vulkan=False) as cpu_engine:
        cpu_engine.read(rgb)
        start = time.perf_counter()
        for _ in range(3):
            cpu_result = cpu_engine.read(rgb)
        cpu = (time.perf_counter() - start) / 3

    print(
        f"{name}: vulkan {vulkan * 1000:.0f} ms ({len(vulkan_result.lines)} lines) | "
        f"ncnn cpu {cpu * 1000:.0f} ms ({len(cpu_result.lines)} lines)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
