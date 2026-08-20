"""Device policy: hardware only, discrete preferred, software refused."""

import pytest

from vulkanocr.device import HardwareVulkanUnavailableError, select_hardware_device


class FakeInfo:
    def __init__(self, name, kind):
        self._name = name
        self._kind = kind

    def device_name(self):
        return self._name

    def type(self):
        return self._kind


class FakeRuntime:
    def __init__(self, devices):
        self._devices = devices

    def get_gpu_count(self):
        return len(self._devices)

    def get_gpu_info(self, index):
        return self._devices[index]


def test_discrete_wins_over_integrated_regardless_of_order():
    runtime = FakeRuntime([FakeInfo("iGPU", 1), FakeInfo("dGPU", 0)])
    device = select_hardware_device(runtime)
    assert (device.index, device.name, device.kind) == (1, "dGPU", 0)


def test_the_software_rasteriser_is_never_selected():
    runtime = FakeRuntime([FakeInfo("llvmpipe", 3), FakeInfo("iGPU", 1)])
    assert select_hardware_device(runtime).name == "iGPU"


def test_software_only_is_a_refusal_not_a_fallback():
    runtime = FakeRuntime([FakeInfo("llvmpipe", 3)])
    with pytest.raises(HardwareVulkanUnavailableError):
        select_hardware_device(runtime)


def test_no_devices_is_a_refusal():
    with pytest.raises(HardwareVulkanUnavailableError):
        select_hardware_device(FakeRuntime([]))


def test_first_of_equal_ranks_is_deterministic():
    runtime = FakeRuntime([FakeInfo("dGPU-a", 0), FakeInfo("dGPU-b", 0)])
    assert select_hardware_device(runtime).name == "dGPU-a"
