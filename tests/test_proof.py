"""The busy-sampler, proven on counters it can actually read."""

import time

from vulkanocr.device import VulkanDevice
from vulkanocr.proof import (
    ProofDevice,
    ProofResult,
    ProofSample,
    busy_sampler,
    system_busy_result,
)


def test_the_sampler_collects_while_the_block_runs_and_stops_after(tmp_path):
    counter = tmp_path / "gpu_busy_percent"
    counter.write_text("37\n")
    with busy_sampler(paths=[counter]) as samples:
        time.sleep(0.08)
    collected = len(samples[counter])
    assert collected >= 2
    assert set(samples[counter]) == {37}
    # Stopped: nothing more arrives after the block.
    time.sleep(0.05)
    assert len(samples[counter]) == collected


def test_a_counter_that_stops_answering_is_skipped_not_fatal(tmp_path):
    vanishing = tmp_path / "gpu_busy_percent"
    vanishing.write_text("nonsense")
    with busy_sampler(paths=[vanishing]) as samples:
        time.sleep(0.05)
    assert samples[vanishing] == []


def test_result_counts_activity_only_from_selected_devices():
    selected = ProofDevice(1, "RX 6600 XT", 0x1002, 0x73FF)
    other = ProofDevice(0, "610M", 0x1002, 0x164E)
    result = ProofResult(
        provider="fake",
        scope="process",
        selected_devices=(selected,),
        samples=(
            ProofSample(other, "drm-engine-compute", 900),
            ProofSample(selected, "drm-engine-compute", 0),
        ),
        supported=True,
    )

    assert result.matching_samples == (ProofSample(selected, "drm-engine-compute", 0),)
    assert result.activity_observed is False


def test_unsupported_result_requires_an_explicit_reason():
    selected = ProofDevice(1, "GPU")

    result = ProofResult(
        provider="fake",
        scope="system",
        selected_devices=(selected,),
        samples=(),
        supported=False,
        unavailable_reason="driver exposes no telemetry counter",
    )

    assert result.supported is False
    assert result.unavailable_reason == "driver exposes no telemetry counter"


def test_missing_busy_counter_returns_explicit_unavailable_state():
    device = VulkanDevice(0, "Intel Arc", 0, 0x8086, 0x56A0)

    result = system_busy_result((device,), {})

    assert result.supported is False
    assert result.samples == ()
    assert result.unavailable_reason == "selected device exposes no gpu_busy_percent counter"


def test_busy_counter_for_another_device_cannot_count_as_selected_activity(tmp_path):
    counter = tmp_path / "card0/device/gpu_busy_percent"
    counter.parent.mkdir(parents=True)
    counter.write_text("91\n")
    (counter.parent / "vendor").write_text("0x1002\n")
    (counter.parent / "device").write_text("0x164e\n")
    selected = VulkanDevice(1, "RX 6600 XT", 0, 0x1002, 0x73FF)

    result = system_busy_result((selected,), {counter: [91]})

    assert result.supported is False
    assert result.activity_observed is False
    assert result.unavailable_reason == "no gpu_busy_percent counter matches the selected device"
