"""The one checked ncnn extractor lifecycle used by every inference stage."""

from __future__ import annotations

from typing import Any


def extract_output(
    net: Any,
    value: Any,
    blobs: tuple[str, str],
    *,
    stage: str,
) -> Any:
    """Submit one value and return one output, releasing the extractor always."""
    extractor = net.create_extractor()
    try:
        extractor.input(blobs[0], value)
        code, output = extractor.extract(blobs[1])
        if code != 0:
            raise RuntimeError(f"{stage} extraction failed")
        return output
    finally:
        del extractor


__all__ = ["extract_output"]
