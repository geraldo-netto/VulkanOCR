"""Proof a read ran on hardware: the kernel's own busy counters.

The AMD DRM counter ``gpu_busy_percent`` is sampled from sysfs while work
runs; a software (llvmpipe) run cannot move it. Instrumentation, not OCR —
which is why it lives here and not in the CLI that happens to print it
(VOCR-0053).
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

SAMPLE_INTERVAL_S = 0.02


def gpu_busy_paths() -> list[Path]:
    return sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))


def _sample(stop: threading.Event, samples: dict[Path, list[int]]) -> None:
    paths = list(samples)
    while not stop.is_set():
        for path in paths:
            # A card that stops answering mid-sample is not worth ending a
            # read over: the counter is an observation, not the work.
            with contextlib.suppress(OSError, ValueError):
                samples[path].append(int(path.read_text().strip()))
        time.sleep(SAMPLE_INTERVAL_S)


@contextlib.contextmanager
def busy_sampler(paths: Sequence[Path] | None = None) -> Iterator[dict[Path, list[int]]]:
    """Sample every card's busy counter for the duration of the block."""
    samples: dict[Path, list[int]] = {
        path: [] for path in (gpu_busy_paths() if paths is None else paths)
    }
    stop = threading.Event()
    sampler = threading.Thread(target=_sample, args=(stop, samples), daemon=True)
    sampler.start()
    try:
        yield samples
    finally:
        stop.set()
        sampler.join(timeout=1)


__all__ = ["busy_sampler", "gpu_busy_paths"]
