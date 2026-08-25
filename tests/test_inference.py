"""Shared ncnn extractor lifecycle without loading the native runtime."""

import gc
import weakref

import pytest

from vulkanocr.inference import extract_output


class FakeNet:
    def __init__(self, code=0, error=None):
        self.code = code
        self.error = error
        self.calls = []
        self.extractor_ref = None

    def create_extractor(self):
        net = self

        class Extractor:
            def input(self, name, value):
                net.calls.append(("input", name, value))

            def extract(self, name):
                net.calls.append(("extract", name))
                if net.error is not None:
                    raise ValueError(net.error)
                return net.code, "output-value"

        extractor = Extractor()
        self.extractor_ref = weakref.ref(extractor)
        return extractor


def test_shared_extractor_submits_extracts_and_releases():
    net = FakeNet()

    output = extract_output(net, "input-value", ("in", "out"), stage="recognition")
    gc.collect()

    assert output == "output-value"
    assert net.calls == [("input", "in", "input-value"), ("extract", "out")]
    assert net.extractor_ref is not None and net.extractor_ref() is None


def test_nonzero_extraction_code_is_checked_and_still_releases():
    net = FakeNet(code=-1)

    with pytest.raises(RuntimeError, match="detection extraction failed"):
        extract_output(net, "input-value", ("in", "out"), stage="detection")
    gc.collect()

    assert net.extractor_ref is not None and net.extractor_ref() is None


def test_native_extraction_exception_still_releases():
    net = FakeNet(error="native failure")

    with pytest.raises(ValueError, match="native failure"):
        extract_output(net, "input-value", ("in", "out"), stage="recognition")
    gc.collect()

    assert net.extractor_ref is not None and net.extractor_ref() is None
