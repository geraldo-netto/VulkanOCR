"""Hardware-only Vulkan device selection.

Mirrors the service's no-CPU rule: a software rasteriser (ncnn device type 3,
llvmpipe) is never selected — if it is the only Vulkan device, the engine
reports unavailable instead of silently running on the host CPU. Discrete
devices are preferred over integrated, integrated over virtual.
"""

from __future__ import annotations

from dataclasses import dataclass

# ncnn VkGpuInfo.type(): 0 discrete, 1 integrated, 2 virtual, 3 cpu (software).
_PREFERENCE = {0: 0, 1: 1, 2: 2}


class HardwareVulkanUnavailable(RuntimeError):
    """No hardware Vulkan device is present; the engine refuses to run."""


@dataclass(frozen=True, slots=True)
class VulkanDevice:
    index: int
    name: str
    kind: int


def select_hardware_device(runtime) -> VulkanDevice:
    """The most capable hardware Vulkan device, or a refusal.

    ``runtime`` is the imported ``ncnn`` module; it is injected so tests can
    substitute a fake without a GPU or the wheel.
    """
    candidates: list[tuple[int, int, VulkanDevice]] = []
    for index in range(runtime.get_gpu_count()):
        info = runtime.get_gpu_info(index)
        kind = info.type()
        rank = _PREFERENCE.get(kind)
        if rank is None:
            continue
        candidates.append((rank, index, VulkanDevice(index, info.device_name(), kind)))
    if not candidates:
        raise HardwareVulkanUnavailable(
            "no hardware Vulkan device is present; refusing the software rasteriser"
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]
