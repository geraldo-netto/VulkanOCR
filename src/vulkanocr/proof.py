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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .device import VulkanDevice

SAMPLE_INTERVAL_S = 0.02
ProofScope = Literal["process", "system"]


@dataclass(frozen=True, slots=True)
class ProofDevice:
    """Stable device facts shared by runtime selection and telemetry."""

    runtime_index: int | None
    name: str
    vendor_id: int | None = None
    device_id: int | None = None
    drm_node: str | None = None

    @classmethod
    def from_vulkan(cls, device: VulkanDevice) -> ProofDevice:
        return cls(device.index, device.name, device.vendor_id, device.device_id)

    def matches(self, other: ProofDevice) -> bool:
        if self.drm_node is not None and other.drm_node is not None:
            return self.drm_node == other.drm_node
        own_pci = (self.vendor_id, self.device_id)
        other_pci = (other.vendor_id, other.device_id)
        if None not in own_pci and None not in other_pci:
            return own_pci == other_pci
        return self.name == other.name


@dataclass(frozen=True, slots=True)
class ProofSample:
    device: ProofDevice
    metric: str
    value: float


@dataclass(frozen=True, slots=True)
class ProofResult:
    """One provider's attributable telemetry result."""

    provider: str
    scope: ProofScope
    selected_devices: tuple[ProofDevice, ...]
    samples: tuple[ProofSample, ...]
    supported: bool
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.selected_devices:
            raise ValueError("proof requires at least one selected device")
        if self.supported and self.unavailable_reason is not None:
            raise ValueError("supported proof cannot carry an unavailable reason")
        if not self.supported and not self.unavailable_reason:
            raise ValueError("unsupported proof requires an unavailable reason")

    @property
    def matching_samples(self) -> tuple[ProofSample, ...]:
        return tuple(
            sample
            for sample in self.samples
            if any(selected.matches(sample.device) for selected in self.selected_devices)
        )

    @property
    def activity_observed(self) -> bool:
        return self.supported and any(sample.value > 0 for sample in self.matching_samples)


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


__all__ = [
    "ProofDevice",
    "ProofResult",
    "ProofSample",
    "ProofScope",
    "busy_sampler",
    "gpu_busy_paths",
]
