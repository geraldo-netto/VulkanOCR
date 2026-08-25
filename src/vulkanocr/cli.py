"""MVP demo: read text from an image on the Vulkan GPU, prove where it ran.

Proof of GPU execution is direct: while recognition loops, the AMD DRM
counter ``gpu_busy_percent`` of the selected card is sampled from sysfs
(:mod:`vulkanocr.proof`). A software (llvmpipe) run cannot move that counter.
"""

from __future__ import annotations

import argparse
import statistics
import time

import cv2

from vulkanocr import (
    CATALOG,
    DEFAULT_MODEL,
    PRECISIONS,
    OcrEngine,
    models_for,
    options_for_precision,
)
from vulkanocr.device import HardwareVulkanUnavailableError
from vulkanocr.engine import OcrEngineError
from vulkanocr.proof import busy_sampler


def _reader(arguments):
    """The engine the flags ask for: one device, or one process per device."""
    options = options_for_precision(arguments.precision)
    if arguments.all_gpus:
        from .parallel import ParallelOcr  # noqa: PLC0415 - spawns worker processes

        return ParallelOcr(models_for(arguments.models), options=options)
    return OcrEngine(models_for(arguments.models), options=options)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="extra timed passes for benchmarking (default: none — one read answers)",
    )
    parser.add_argument(
        "--all-gpus",
        action="store_true",
        help="recognise crops on every hardware Vulkan device, not just the best one",
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODEL,
        choices=sorted(CATALOG),
        help=f"model set to run (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--precision",
        choices=PRECISIONS,
        default="fp32",
        help="ncnn execution precision (default: fp32)",
    )
    return parser.parse_args(argv)


def main() -> int:
    arguments = _arguments()
    bgr = cv2.imread(arguments.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read image: {arguments.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # A person at a terminal gets the refusal, not the machinery around it:
    # a missing model, a missing ncnn wheel and a machine with no hardware
    # Vulkan device are all states with a next step, and a traceback buries
    # the sentence that names it.
    try:
        engine = _reader(arguments)
    except (OcrEngineError, HardwareVulkanUnavailableError) as error:
        raise SystemExit(str(error)) from error
    print(f"models: {arguments.models} — {CATALOG[arguments.models].note}")
    print(f"precision: {arguments.precision}")
    print(f"device: {engine.device_name}")
    with engine, busy_sampler() as samples:
        result, first_ms, timings = _timed_reads(engine, rgb, arguments.repeat)
    _report(result, first_ms, timings, samples)
    return 0


def _timed_reads(engine, rgb, repeats: int):
    """One answering read and any number of timed extras."""
    started = time.monotonic()
    result = engine.read(rgb)
    first_ms = (time.monotonic() - started) * 1000
    timings = []
    for _ in range(repeats):
        started = time.monotonic()
        engine.read(rgb)
        timings.append((time.monotonic() - started) * 1000)
    return result, first_ms, timings


def _report(result, first_ms: float, timings: list[float], samples) -> None:
    """Say what was read, how fast, and which silicon was busy doing it."""
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
    if result.undecoded_regions:
        # Without this line a clean page and a page the recogniser gave up
        # on printed identically — the exact invisibility the field's own
        # docstring warns about (VOCR-0045).
        print(f"{result.undecoded_regions} detected regions could not be decoded")

    print("\nGPU busy while reading (sysfs gpu_busy_percent):")
    for path, values in samples.items():
        if values:
            print(
                f"  {path.parent.parent.name}: max {max(values)}%  "
                f"mean {statistics.fmean(values):.1f}%  samples {len(values)}"
            )


if __name__ == "__main__":
    import sys

    sys.exit(main())
