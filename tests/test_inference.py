"""Shared ncnn extractor lifecycle without loading the native runtime."""

import gc
import weakref

import pytest

from vulkanocr.inference import NcnnInferenceError, extract_output


class FakeNet:
    def __init__(self, *, input_code=0, extract_code=0, failure=None):
        self.input_code = input_code
        self.extract_code = extract_code
        self.failure = failure
        self.calls = []
        self.extractor_ref = None

    def create_extractor(self):
        if self.failure == "create-extractor":
            raise ValueError("native create failure")
        net = self

        class Extractor:
            def input(self, name, value):
                net.calls.append(("input", name, value))
                if net.failure == "input":
                    raise ValueError("native input failure")
                return net.input_code

            def extract(self, name):
                net.calls.append(("extract", name))
                if net.failure == "extract":
                    raise ValueError("native extract failure")
                return net.extract_code, "output-value"

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
    net = FakeNet(extract_code=-1)

    with pytest.raises(NcnnInferenceError, match="extract failed: ncnn returned -1") as caught:
        extract_output(net, "input-value", ("in", "out"), stage="detection")
    gc.collect()

    assert (caught.value.stage, caught.value.operation) == ("detection", "extract")
    assert net.extractor_ref is not None and net.extractor_ref() is None


def test_nonzero_input_code_is_checked_before_extraction():
    net = FakeNet(input_code=-2)

    with pytest.raises(NcnnInferenceError, match="input failed: ncnn returned -2") as caught:
        extract_output(net, "input-value", ("in", "out"), stage="recognition")
    gc.collect()

    assert (caught.value.stage, caught.value.operation) == ("recognition", "input")
    assert net.calls == [("input", "in", "input-value")]
    assert net.extractor_ref is not None and net.extractor_ref() is None


@pytest.mark.parametrize("operation", ["input", "extract"])
def test_native_operation_exceptions_are_translated_and_release(operation):
    net = FakeNet(failure=operation)

    with pytest.raises(NcnnInferenceError, match=f"{operation} failed: native") as caught:
        extract_output(net, "input-value", ("in", "out"), stage="orientation")
    gc.collect()

    assert (caught.value.stage, caught.value.operation) == ("orientation", operation)
    assert net.extractor_ref is not None and net.extractor_ref() is None


def test_extractor_creation_exception_is_translated_without_cleanup_target():
    net = FakeNet(failure="create-extractor")

    with pytest.raises(NcnnInferenceError, match="create-extractor failed") as caught:
        extract_output(net, "input-value", ("in", "out"), stage="detection")

    assert (caught.value.stage, caught.value.operation) == ("detection", "create-extractor")
    assert net.extractor_ref is None
