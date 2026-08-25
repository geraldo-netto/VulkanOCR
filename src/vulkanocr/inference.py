"""The one checked ncnn extractor lifecycle used by every inference stage."""

from __future__ import annotations

from typing import Any


class NcnnInferenceError(RuntimeError):
    """One native extractor operation failed for a named OCR stage."""

    def __init__(self, stage: str, operation: str, detail: str):
        self.stage = stage
        self.operation = operation
        self.detail = detail
        super().__init__(f"{operation} failed: {detail}")


def _native_detail(error: Exception) -> str:
    return str(error) or type(error).__name__


def _native_failure(stage: str, operation: str, error: Exception) -> NcnnInferenceError:
    detail = _native_detail(error)
    # Native exceptions capture the extractor as a traceback local. Keeping
    # that chain would defeat the finally block until the caller released the
    # translated error, so retain its useful text but release its traceback.
    error.__traceback__ = None
    return NcnnInferenceError(stage, operation, detail)


def _native_call(function: Any, *args: Any) -> tuple[bool, Any]:
    try:
        return True, function(*args)
    except Exception as error:  # noqa: BLE001 - native binding has no exception taxonomy
        detail = _native_detail(error)
        error.__traceback__ = None
        return False, detail


def extract_output(
    net: Any,
    value: Any,
    blobs: tuple[str, str],
    *,
    stage: str,
) -> Any:
    """Submit one value and return one output, releasing the extractor always."""
    succeeded, extractor = _native_call(net.create_extractor)
    if not succeeded:
        raise NcnnInferenceError(stage, "create-extractor", extractor)
    try:
        succeeded, input_code = _native_call(extractor.input, blobs[0], value)
        if not succeeded:
            raise NcnnInferenceError(stage, "input", input_code)
        # Some binding versions expose the void C++ overload as None.
        if input_code not in (None, 0):
            raise NcnnInferenceError(stage, "input", f"ncnn returned {input_code}")
        succeeded, result = _native_call(extractor.extract, blobs[1])
        if not succeeded:
            raise NcnnInferenceError(stage, "extract", result)
        try:
            code, output = result
        except (TypeError, ValueError) as error:
            raise _native_failure(stage, "extract", error) from None
        if code != 0:
            raise NcnnInferenceError(stage, "extract", f"ncnn returned {code}")
        return output
    finally:
        del extractor


__all__ = ["NcnnInferenceError", "extract_output"]
