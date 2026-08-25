"""Shared benchmark lifecycle and precondition behavior."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import benchmark_harness  # noqa: E402
from benchmark_harness import benchmark_engine, require_nonempty, require_text_crops  # noqa: E402
from strips import pack  # noqa: E402


def test_engine_is_closed_when_measurement_raises(monkeypatch):
    class FakeEngine:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_error):
            self.closed = True

    engine = FakeEngine()
    monkeypatch.setattr(benchmark_harness, "models_for", lambda _name: object())
    monkeypatch.setattr(benchmark_harness, "OcrEngine", lambda _models, **_options: engine)

    with pytest.raises(RuntimeError, match="measurement failed"), benchmark_engine("fake"):
        raise RuntimeError("measurement failed")

    assert engine.closed is True


def test_empty_benchmark_input_is_a_clear_refusal():
    with pytest.raises(SystemExit, match="benchmark refused: no text crops"):
        require_nonempty([], "text crops detected")


def test_crop_precondition_is_shared_by_batching_scripts():
    class EmptyEngine:
        def crops(self, rgb):
            del rgb
            return []

    with pytest.raises(SystemExit, match="benchmark refused: no text crops"):
        require_text_crops(EmptyEngine(), object())


def test_strip_packer_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="empty crop sequence"):
        pack([])
