"""Does a second GPU actually speed a page? Measured, either way.

One queue, one engine per device: the fast card takes crops as fast as it
finishes them, the slow one contributes what it can. Whether that beats the
fast card alone depends on how asymmetric the pair is and on how much of the
binding runs outside the GIL — both are this script's job to answer, not
assume.
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from benchmark_harness import benchmark_engine
from scoring import load_rgb

from vulkanocr.catalog import models_for
from vulkanocr.device import hardware_devices
from vulkanocr.parallel import ParallelOcr

RUNS = 5


def timed(reader, rgb):
    reader.read(rgb)  # warm
    start = time.perf_counter()
    for _ in range(RUNS):
        result = reader.read(rgb)
    return (time.perf_counter() - start) / RUNS * 1000, result


def main() -> int:
    import ncnn

    rgb = load_rgb(sys.argv[1])
    model_set = sys.argv[2] if len(sys.argv) > 2 else "v6-medium"
    devices = hardware_devices(ncnn)
    print("hardware devices:", [d.name for d in devices])

    with benchmark_engine(model_set) as single:
        single_ms, single_result = timed(single, rgb)
        print(
            f"single ({single.device_name}): {single_ms:.0f} ms, "
            f"{len(single_result.lines)} lines"
        )
        single_texts = [line.text for line in single_result.lines]

    if len(devices) < 2:
        print("one hardware device; nothing to parallelise")
        return 0

    with ParallelOcr(models_for(model_set)) as pool:
        pool_ms, pool_result = timed(pool, rgb)
        print(
            f"pool ({' + '.join(pool.device_names)}): {pool_ms:.0f} ms, "
            f"{len(pool_result.lines)} lines"
        )
        identical = [line.text for line in pool_result.lines] == single_texts
        print(f"speedup {single_ms / pool_ms:.2f}x   text identical: {identical}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
