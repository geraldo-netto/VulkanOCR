"""MVP demo: read text from an image on Vulkan and report attributable telemetry.

Per-process DRM fdinfo engine counters are preferred. AMD's
``gpu_busy_percent`` is a clearly labeled system-wide fallback; unsupported
drivers and mismatched devices produce an explicit unavailable state.
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
from vulkanocr.proof import ProofResult, proof_sampler


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


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
        type=_non_negative_int,
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

    # A person at a terminal gets the refusal, not the machinery around it.
    # Reads share the construction boundary so a native failure is concise
    # and the engine's context manager still releases every loaded network.
    try:
        engine = _reader(arguments)
        print(f"models: {arguments.models} — {CATALOG[arguments.models].note}")
        print(f"precision: {arguments.precision}")
        print(f"device: {engine.device_name}")
        with engine, proof_sampler(engine.devices) as proof_capture:
            result, first_ms, timings = _timed_reads(engine, rgb, arguments.repeat)
        proof = proof_capture.finished_result()
    except (OcrEngineError, HardwareVulkanUnavailableError) as error:
        raise SystemExit(str(error)) from error
    _report(result, first_ms, timings, proof)
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


def _report(result, first_ms: float, timings: list[float], proof: ProofResult) -> None:
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
    if result.filtered_regions:
        print(f"{result.filtered_regions} recognised regions were filtered as known noise")

    _report_proof(proof)


def _report_proof(proof: ProofResult) -> None:
    if not proof.supported:
        print(f"\nGPU telemetry unavailable: {proof.unavailable_reason}")
        return
    print(f"\nGPU activity ({proof.provider}, {proof.scope}-wide):")
    if proof.scope == "process":
        totals: dict[str, float] = {}
        for sample in proof.matching_samples:
            totals[sample.metric] = totals.get(sample.metric, 0.0) + sample.value
        for metric, nanoseconds in sorted(totals.items()):
            print(f"  {metric}: +{nanoseconds / 1e6:.1f} ms")
        return
    values = [sample.value for sample in proof.matching_samples]
    print(
        f"  max {max(values):.0f}%  mean {statistics.fmean(values):.1f}%  "
        f"samples {len(values)}"
    )


if __name__ == "__main__":
    import sys

    sys.exit(main())
