"""Proof a read ran on hardware: the kernel's own busy counters.

The AMD DRM counter ``gpu_busy_percent`` is sampled from sysfs while work
runs; a software (llvmpipe) run cannot move it. Instrumentation, not OCR —
which is why it lives here and not in the CLI that happens to print it
(VOCR-0053).
"""

from __future__ import annotations

import contextlib
import re
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .device import VulkanDevice

SAMPLE_INTERVAL_S = 0.02
ProofScope = Literal["process", "system"]
FdinfoSnapshot = dict[tuple[str, str, str], tuple["ProofDevice", dict[str, int]]]


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


@dataclass(slots=True)
class ProofCapture:
    result: ProofResult | None = None

    def finished_result(self) -> ProofResult:
        if self.result is None:
            raise RuntimeError("proof capture has not finished")
        return self.result


def gpu_busy_paths() -> list[Path]:
    return sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))


def _hex_attribute(path: Path, name: str) -> int | None:
    try:
        return int((path.parent / name).read_text().strip(), 16)
    except (OSError, ValueError):
        return None


def _busy_device(path: Path) -> ProofDevice:
    node = path.parent.parent.name
    return ProofDevice(
        runtime_index=None,
        name=node,
        vendor_id=_hex_attribute(path, "vendor"),
        device_id=_hex_attribute(path, "device"),
        drm_node=node,
    )


def system_busy_result(
    devices: Sequence[VulkanDevice], samples: dict[Path, list[int]]
) -> ProofResult:
    """Convert AMD's system-wide counter samples into attributable state."""
    selected = tuple(ProofDevice.from_vulkan(device) for device in devices)
    if not samples:
        return ProofResult(
            provider="amd-gpu-busy-percent",
            scope="system",
            selected_devices=selected,
            samples=(),
            supported=False,
            unavailable_reason="selected device exposes no gpu_busy_percent counter",
        )
    path_devices = {path: _busy_device(path) for path in samples}
    matching_paths = {
        path
        for path, observed in path_devices.items()
        if any(device.matches(observed) for device in selected)
    }
    proof_samples = tuple(
        ProofSample(path_devices[path], "gpu_busy_percent", value)
        for path, values in samples.items()
        for value in values
    )
    if not matching_paths:
        return ProofResult(
            provider="amd-gpu-busy-percent",
            scope="system",
            selected_devices=selected,
            samples=proof_samples,
            supported=False,
            unavailable_reason="no gpu_busy_percent counter matches the selected device",
        )
    identity_samples = tuple(
        ProofSample(device, "gpu_busy_percent", 0) for device in path_devices.values()
    )
    if _has_ambiguous_pci_identity(selected, identity_samples):
        return ProofResult(
            provider="amd-gpu-busy-percent",
            scope="system",
            selected_devices=selected,
            samples=proof_samples,
            supported=False,
            unavailable_reason="multiple DRM devices share the selected PCI identity",
        )
    if not any(samples[path] for path in matching_paths):
        return ProofResult(
            provider="amd-gpu-busy-percent",
            scope="system",
            selected_devices=selected,
            samples=proof_samples,
            supported=False,
            unavailable_reason="gpu_busy_percent counter produced no readable samples",
        )
    candidate = ProofResult(
        provider="amd-gpu-busy-percent",
        scope="system",
        selected_devices=selected,
        samples=proof_samples,
        supported=True,
    )
    return candidate


def _has_ambiguous_pci_identity(
    selected: tuple[ProofDevice, ...], samples: tuple[ProofSample, ...]
) -> bool:
    for device in selected:
        identity = (device.vendor_id, device.device_id)
        if None in identity:
            continue
        selected_count = sum(
            (item.vendor_id, item.device_id) == identity for item in selected
        )
        nodes = {
            sample.device.drm_node
            for sample in samples
            if (sample.device.vendor_id, sample.device.device_id) == identity
        }
        if len(nodes) > selected_count:
            return True
    return False


def drm_fdinfo_snapshot(
    fdinfo_root: Path = Path("/proc/self/fdinfo"),
    pci_root: Path = Path("/sys/bus/pci/devices"),
) -> FdinfoSnapshot:
    """Read one de-duplicated snapshot of per-process DRM engine counters."""
    clients: FdinfoSnapshot = {}
    try:
        paths = tuple(fdinfo_root.iterdir())
    except OSError:
        return clients
    for path in paths:
        try:
            text = path.read_text()
        except OSError:
            continue
        driver = _fdinfo_field(text, "drm-driver")
        client = _fdinfo_field(text, "drm-client-id")
        pdev = _fdinfo_field(text, "drm-pdev")
        if driver is None or client is None or pdev is None:
            continue
        counters = {
            metric: int(value)
            for metric, value in re.findall(r"^(drm-engine-[^:]+):\s+(\d+)\s+ns", text, re.M)
        }
        if not counters:
            continue
        device_path = pci_root / pdev
        device = ProofDevice(
            runtime_index=None,
            name=f"{driver}:{pdev}",
            vendor_id=_read_hex(device_path / "vendor"),
            device_id=_read_hex(device_path / "device"),
            drm_node=pdev,
        )
        key = (driver, client, pdev)
        if key not in clients:
            clients[key] = device, counters
        else:
            previous_device, previous = clients[key]
            clients[key] = previous_device, {
                metric: max(previous.get(metric, 0), counters.get(metric, 0))
                for metric in set(previous) | set(counters)
            }
    return clients


def _fdinfo_field(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(\S+)", text, re.M)
    return match.group(1) if match is not None else None


def _read_hex(path: Path) -> int | None:
    try:
        return int(path.read_text().strip(), 16)
    except (OSError, ValueError):
        return None


def process_fdinfo_result(
    devices: Sequence[VulkanDevice],
    before: FdinfoSnapshot,
    after: FdinfoSnapshot,
) -> ProofResult:
    """Return per-process engine-time deltas attributable to selected devices."""
    selected = tuple(ProofDevice.from_vulkan(device) for device in devices)
    samples = []
    for key in sorted(set(before) | set(after)):
        entry = after.get(key) or before.get(key)
        if entry is None:
            continue
        device, _counters = entry
        old = before.get(key, (device, {}))[1]
        new = after.get(key, (device, {}))[1]
        for metric in sorted(set(old) | set(new)):
            delta = max(0, new.get(metric, 0) - old.get(metric, 0))
            samples.append(ProofSample(device, metric, delta))
    candidate = ProofResult(
        provider="drm-fdinfo",
        scope="process",
        selected_devices=selected,
        samples=tuple(samples),
        supported=True,
    )
    if not candidate.matching_samples:
        return ProofResult(
            provider=candidate.provider,
            scope=candidate.scope,
            selected_devices=selected,
            samples=candidate.samples,
            supported=False,
            unavailable_reason="DRM fdinfo exposes no engine counters for the selected device",
        )
    if _has_ambiguous_pci_identity(selected, candidate.samples):
        return ProofResult(
            provider=candidate.provider,
            scope=candidate.scope,
            selected_devices=selected,
            samples=candidate.samples,
            supported=False,
            unavailable_reason="multiple DRM devices share the selected PCI identity",
        )
    return candidate


def preferred_proof_result(process: ProofResult, system: ProofResult) -> ProofResult:
    """Prefer attributable process counters; AMD system telemetry is fallback."""
    if process.supported:
        return process
    if system.supported:
        return system
    return ProofResult(
        provider="none",
        scope="process",
        selected_devices=process.selected_devices,
        samples=(),
        supported=False,
        unavailable_reason=f"{process.unavailable_reason}; {system.unavailable_reason}",
    )


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


@contextlib.contextmanager
def proof_sampler(
    devices: Sequence[VulkanDevice],
    *,
    fdinfo_root: Path = Path("/proc/self/fdinfo"),
    pci_root: Path = Path("/sys/bus/pci/devices"),
    busy_paths: Sequence[Path] | None = None,
) -> Iterator[ProofCapture]:
    """Capture preferred per-process proof and AMD's system-wide fallback."""
    before = drm_fdinfo_snapshot(fdinfo_root, pci_root)
    capture = ProofCapture()
    with busy_sampler(busy_paths) as busy_samples:
        try:
            yield capture
        finally:
            after = drm_fdinfo_snapshot(fdinfo_root, pci_root)
            process = process_fdinfo_result(devices, before, after)
            system = system_busy_result(devices, busy_samples)
            capture.result = preferred_proof_result(process, system)


__all__ = [
    "ProofDevice",
    "ProofCapture",
    "ProofResult",
    "ProofSample",
    "ProofScope",
    "busy_sampler",
    "drm_fdinfo_snapshot",
    "gpu_busy_paths",
    "preferred_proof_result",
    "process_fdinfo_result",
    "proof_sampler",
    "system_busy_result",
]
