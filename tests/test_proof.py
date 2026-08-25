"""The busy-sampler, proven on counters it can actually read."""

import time

import pytest

from vulkanocr.device import VulkanDevice
from vulkanocr.proof import (
    ProofDevice,
    ProofResult,
    ProofSample,
    busy_sampler,
    drm_fdinfo_snapshot,
    preferred_proof_result,
    process_fdinfo_result,
    proof_sampler,
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


@pytest.mark.parametrize(
    ("driver", "vendor", "device_id", "metric"),
    [
        ("amdgpu", 0x1002, 0x73FF, "drm-engine-compute"),
        ("i915", 0x8086, 0x56A0, "drm-engine-render"),
    ],
)
def test_amd_and_intel_fdinfo_engine_counters_are_supported(
    tmp_path, driver, vendor, device_id, metric
):
    fdinfo = tmp_path / "fdinfo"
    pci = tmp_path / "pci"
    fdinfo.mkdir()
    pdev = "0000:03:00.0"
    device_path = pci / pdev
    device_path.mkdir(parents=True)
    (device_path / "vendor").write_text(f"0x{vendor:04x}\n")
    (device_path / "device").write_text(f"0x{device_id:04x}\n")
    record = fdinfo / "5"
    prefix = (
        f"drm-driver:\t{driver}\n"
        f"drm-client-id:\t42\n"
        f"drm-pdev:\t{pdev}\n"
    )
    record.write_text(f"{prefix}{metric}:\t100 ns\n")
    selected = VulkanDevice(0, "Selected GPU", 0, vendor, device_id)

    with proof_sampler(
        (selected,), fdinfo_root=fdinfo, pci_root=pci, busy_paths=[]
    ) as capture:
        record.write_text(f"{prefix}{metric}:\t700 ns\n")

    result = capture.finished_result()
    assert result.provider == "drm-fdinfo"
    assert result.scope == "process"
    assert result.activity_observed is True
    assert result.matching_samples == (
        ProofSample(result.matching_samples[0].device, metric, 600),
    )


def test_amd_busy_percent_is_used_as_system_fallback(tmp_path):
    fdinfo = tmp_path / "fdinfo"
    fdinfo.mkdir()
    counter = tmp_path / "drm/card0/device/gpu_busy_percent"
    counter.parent.mkdir(parents=True)
    counter.write_text("63\n")
    (counter.parent / "vendor").write_text("0x1002\n")
    (counter.parent / "device").write_text("0x73ff\n")
    selected = VulkanDevice(0, "RX 6600 XT", 0, 0x1002, 0x73FF)

    with proof_sampler((selected,), fdinfo_root=fdinfo, busy_paths=[counter]) as capture:
        time.sleep(0.05)

    result = capture.finished_result()
    assert result.provider == "amd-gpu-busy-percent"
    assert result.scope == "system"
    assert result.activity_observed is True


def test_nvidia_without_engine_or_busy_counters_is_explicitly_unavailable(tmp_path):
    fdinfo = tmp_path / "fdinfo"
    fdinfo.mkdir()
    (fdinfo / "9").write_text(
        "drm-driver:\tnvidia\n"
        "drm-client-id:\t7\n"
        "drm-pdev:\t0000:01:00.0\n"
        "drm-memory-vram:\t1024 KiB\n"
    )
    selected = VulkanDevice(0, "NVIDIA GPU", 0, 0x10DE, 0x2684)

    with proof_sampler((selected,), fdinfo_root=fdinfo, busy_paths=[]) as capture:
        pass

    result = capture.finished_result()
    assert result.provider == "none"
    assert result.supported is False
    assert result.unavailable_reason is not None
    assert "no engine counters" in result.unavailable_reason
    assert "no gpu_busy_percent" in result.unavailable_reason


def test_matching_counter_that_cannot_be_read_is_unavailable(tmp_path):
    counter = tmp_path / "card0/device/gpu_busy_percent"
    counter.parent.mkdir(parents=True)
    (counter.parent / "vendor").write_text("0x1002\n")
    (counter.parent / "device").write_text("0x73ff\n")
    selected = VulkanDevice(0, "RX 6600 XT", 0, 0x1002, 0x73FF)

    result = system_busy_result((selected,), {counter: []})

    assert result.supported is False
    assert result.unavailable_reason == "gpu_busy_percent counter produced no readable samples"


def test_wrong_device_fdinfo_cannot_prove_selected_device(tmp_path):
    fdinfo = tmp_path / "fdinfo"
    pci = tmp_path / "pci"
    fdinfo.mkdir()
    other_path = pci / "0000:03:00.0"
    other_path.mkdir(parents=True)
    (other_path / "vendor").write_text("0x8086\n")
    (other_path / "device").write_text("0x56a0\n")
    (fdinfo / "5").write_text(
        "drm-driver:\ti915\n"
        "drm-client-id:\t42\n"
        "drm-pdev:\t0000:03:00.0\n"
        "drm-engine-render:\t700 ns\n"
    )
    snapshot = drm_fdinfo_snapshot(fdinfo, pci)
    selected = VulkanDevice(0, "RX 6600 XT", 0, 0x1002, 0x73FF)

    result = process_fdinfo_result((selected,), {}, snapshot)

    assert result.supported is False
    assert result.activity_observed is False
    assert result.unavailable_reason == (
        "DRM fdinfo exposes no engine counters for the selected device"
    )
