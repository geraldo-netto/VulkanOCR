"""CLI option wiring without loading a model or touching a GPU."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from vulkanocr import InferenceOptions, cli
from vulkanocr.engine import OcrEngineError


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
    monkeypatch.setattr(cli, "busy_sampler", lambda: nullcontext({}))

    with pytest.raises(SystemExit, match="recognition-inference-failed: extract failed"):
        cli.main()

    assert engine.closed is True
