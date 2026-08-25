"""Hardware-only Vulkan device selection.

Mirrors the service's no-CPU rule: a software rasteriser (ncnn device type 3,
llvmpipe) is never selected — if it is the only Vulkan device, the engine
reports unavailable instead of silently running on the host CPU. Discrete
devices are preferred over integrated, integrated over virtual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

# ncnn VkGpuInfo.type(): 0 discrete, 1 integrated, 2 virtual, 3 cpu (software).
_PREFERENCE = {0: 0, 1: 1, 2: 2}


class HardwareVulkanUnavailableError(RuntimeError):
    """No hardware Vulkan device is present; the engine refuses to run."""


@dataclass(frozen=True, slots=True)
class VulkanDevice:
    index: int
    name: str
    kind: int
    vendor_id: int | None = None
    device_id: int | None = None


def _optional_info_int(info, name: str) -> int | None:
    value = getattr(info, name, None)
    if not callable(value):
        return None
    resolved = cast(Any, value)()
    return int(resolved) if resolved is not None else None


def hardware_devices(runtime) -> tuple[VulkanDevice, ...]:
    """Every hardware Vulkan device, most capable first, or a refusal.

    The single-device selection below is its first element; the parallel
    engine takes the whole tuple. Software rasterisers are excluded the same
    way in both — a second "device" that is llvmpipe would be a CPU lane in
    costume.
    """
    candidates: list[tuple[int, int, VulkanDevice]] = []
    for index in range(runtime.get_gpu_count()):
        info = runtime.get_gpu_info(index)
        kind = info.type()
        rank = _PREFERENCE.get(kind)
        if rank is None:
            continue
        candidates.append(
            (
                rank,
                index,
                VulkanDevice(
                    index,
                    info.device_name(),
                    kind,
                    _optional_info_int(info, "vendor_id"),
                    _optional_info_int(info, "device_id"),
                ),
            )
        )
    if not candidates:
        raise HardwareVulkanUnavailableError(
            "no hardware Vulkan device is present; refusing the software rasteriser"
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return tuple(device for _rank, _index, device in candidates)


def select_hardware_device(runtime) -> VulkanDevice:
    """The most capable hardware Vulkan device, or a refusal.

    ``runtime`` is the imported ``ncnn`` module; it is injected so tests can
    substitute a fake without a GPU or the wheel.
    """
    return hardware_devices(runtime)[0]
