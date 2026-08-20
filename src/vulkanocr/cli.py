"""MVP demo: read text from an image on the Vulkan GPU, prove where it ran.

Proof of GPU execution is direct: while recognition loops, the AMD DRM
counter ``gpu_busy_percent`` of the selected card is sampled from sysfs.
A software (llvmpipe) run cannot move that counter.
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import threading
import time
from pathlib import Path

import cv2

from vulkanocr import CATALOG, DEFAULT_MODEL, OcrEngine, models_for
from vulkanocr.device import HardwareVulkanUnavailableError
from vulkanocr.engine import OcrEngineError


def gpu_busy_paths() -> list[Path]:
    return sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))


def sample_gpu_busy(stop: threading.Event, samples: dict[Path, list[int]]) -> None:
    paths = list(samples)
    while not stop.is_set():
        for path in paths:
            # A card that stops answering mid-sample is not worth ending a
            # read over: the counter is an observation, not the work.
            with contextlib.suppress(OSError, ValueError):
                samples[path].append(int(path.read_text().strip()))
        time.sleep(0.02)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="extra timed passes for benchmarking (default: none — one read answers)",
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODEL,
        choices=sorted(CATALOG),
        help=f"model set to run (default: {DEFAULT_MODEL})",
    )
    arguments = parser.parse_args()

    bgr = cv2.imread(arguments.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read image: {arguments.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # A person at a terminal gets the refusal, not the machinery around it:
    # a missing model, a missing ncnn wheel and a machine with no hardware
    # Vulkan device are all states with a next step, and a traceback buries
    # the sentence that names it.
    try:
        engine = OcrEngine(models_for(arguments.models))
    except (OcrEngineError, HardwareVulkanUnavailableError) as error:
        raise SystemExit(str(error)) from error
    print(f"models: {arguments.models} — {CATALOG[arguments.models].note}")
    print(f"device: {engine.device_name}")

    samples: dict[Path, list[int]] = {path: [] for path in gpu_busy_paths()}
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu_busy, args=(stop, samples), daemon=True)
    sampler.start()

    started = time.monotonic()
    result = engine.read(rgb)
    first_ms = (time.monotonic() - started) * 1000

    timings = []
    for _ in range(arguments.repeat):
        started = time.monotonic()
        engine.read(rgb)
        timings.append((time.monotonic() - started) * 1000)

    stop.set()
    sampler.join(timeout=1)

    if timings:
        print(
            f"\nfirst read {first_ms:.0f} ms; warm reads "
            f"median {statistics.median(timings):.0f} ms over {len(timings)} passes\n"
        )
    else:
        print(f"\nread in {first_ms:.0f} ms\n")
    for line in result.lines:
        print(
            f"  ({line.center_x:5.0f},{line.center_y:5.0f}) conf={line.confidence:.2f}  {line.text}"
        )
    print(f"\n{len(result.lines)} lines")

    print("\nGPU busy while reading (sysfs gpu_busy_percent):")
    for path, values in samples.items():
        if values:
            print(
                f"  {path.parent.parent.name}: max {max(values)}%  "
                f"mean {statistics.fmean(values):.1f}%  samples {len(values)}"
            )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
