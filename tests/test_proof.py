"""The busy-sampler, proven on counters it can actually read."""

import time

from vulkanocr.device import VulkanDevice
from vulkanocr.proof import (
    ProofDevice,
    ProofResult,
    ProofSample,
    busy_sampler,
    drm_fdinfo_snapshot,
    preferred_proof_result,
    process_fdinfo_result,
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


def test_drm_fdinfo_returns_selected_process_engine_delta(tmp_path):
    fdinfo = tmp_path / "fdinfo"
    pci = tmp_path / "pci"
    fdinfo.mkdir()
    device_path = pci / "0000:08:00.0"
    device_path.mkdir(parents=True)
    (device_path / "vendor").write_text("0x1002\n")
    (device_path / "device").write_text("0x73ff\n")
    record = fdinfo / "4"
    record.write_text(
        "drm-driver:\tamdgpu\n"
        "drm-client-id:\t17\n"
        "drm-pdev:\t0000:08:00.0\n"
        "drm-engine-compute:\t100 ns\n"
    )
    before = drm_fdinfo_snapshot(fdinfo, pci)
    record.write_text(record.read_text().replace("100 ns", "900 ns"))
    after = drm_fdinfo_snapshot(fdinfo, pci)
    selected = VulkanDevice(1, "RX 6600 XT", 0, 0x1002, 0x73FF)

    result = process_fdinfo_result((selected,), before, after)

    assert result.supported is True
    assert result.scope == "process"
    assert result.provider == "drm-fdinfo"
    assert result.matching_samples[0].value == 800
    assert result.activity_observed is True


def test_supported_system_counter_is_explicit_fallback_for_missing_fdinfo():
    selected = ProofDevice(1, "RX 6600 XT", 0x1002, 0x73FF)
    process = ProofResult(
        "drm-fdinfo",
        "process",
        (selected,),
        (),
        False,
        "no process counters",
    )
    system = ProofResult(
        "amd-gpu-busy-percent",
        "system",
        (selected,),
        (ProofSample(selected, "gpu_busy_percent", 80),),
        True,
    )

    result = preferred_proof_result(process, system)

    assert result is system
    assert result.scope == "system"
