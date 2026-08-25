"""CLI option wiring without loading a model or touching a GPU."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from vulkanocr import InferenceOptions, cli
from vulkanocr.device import VulkanDevice
from vulkanocr.engine import OcrEngineError
from vulkanocr.proof import ProofDevice, ProofResult, ProofSample


def test_precision_defaults_to_fp32():
    arguments = cli._arguments(["page.png"])
    assert arguments.precision == "fp32"


def test_negative_repeat_is_refused():
    with pytest.raises(SystemExit):
        cli._arguments(["page.png", "--repeat", "-1"])


def test_precision_choice_reaches_the_single_engine(monkeypatch):
    received = []
    monkeypatch.setattr(cli, "models_for", lambda _name: object())
    monkeypatch.setattr(
        cli,
        "OcrEngine",
        lambda _models, *, options: received.append(options) or SimpleNamespace(),
    )

    cli._reader(cli._arguments(["page.png", "--precision", "int8"]))

    assert received == [InferenceOptions.int8()]


def test_read_time_engine_error_is_printed_and_the_engine_is_closed(monkeypatch):
    class FailingEngine:
        device_name = "Test GPU"
        devices = (VulkanDevice(0, "Test GPU", 0),)
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_exception):
            self.closed = True

        def read(self, _rgb):
            raise OcrEngineError("recognition-inference-failed", "extract failed")

    engine = FailingEngine()
    arguments = SimpleNamespace(
        image="page.png",
        models="v6-medium",
        precision="fp32",
        repeat=0,
        all_gpus=False,
    )
    monkeypatch.setattr(cli, "_arguments", lambda: arguments)
    monkeypatch.setattr(cli.cv2, "imread", lambda *_args: np.zeros((2, 2, 3), np.uint8))
    monkeypatch.setattr(cli.cv2, "cvtColor", lambda image, _conversion: image)
    monkeypatch.setattr(cli, "_reader", lambda _arguments: engine)
    monkeypatch.setattr(
        cli,
        "proof_sampler",
        lambda _devices: nullcontext(SimpleNamespace(result=None)),
    )

    with pytest.raises(SystemExit, match="recognition-inference-failed: extract failed"):
        cli.main()

    assert engine.closed is True


def test_report_prints_an_explicit_telemetry_unavailable_state(capsys):
    result = SimpleNamespace(lines=(), undecoded_regions=0, filtered_regions=0)
    proof = ProofResult(
        provider="fake",
        scope="system",
        selected_devices=(ProofDevice(0, "Test GPU"),),
        samples=(),
        supported=False,
        unavailable_reason="driver exposes no telemetry counter",
    )

    cli._report(result, 10.0, [], proof)

    output = capsys.readouterr().out
    assert "GPU telemetry unavailable: driver exposes no telemetry counter" in output


def test_report_labels_per_process_engine_time(capsys):
    result = SimpleNamespace(lines=(), undecoded_regions=0, filtered_regions=0)
    device = ProofDevice(0, "Test GPU", 0x1002, 0x73FF)
    proof = ProofResult(
        provider="drm-fdinfo",
        scope="process",
        selected_devices=(device,),
        samples=(ProofSample(device, "drm-engine-compute", 2_500_000),),
        supported=True,
    )

    cli._report(result, 10.0, [], proof)

    output = capsys.readouterr().out
    assert "GPU activity (drm-fdinfo, process-wide)" in output
    assert "drm-engine-compute: +2.5 ms" in output
