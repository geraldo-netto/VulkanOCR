"""Shared lifecycle and input preconditions for standalone benchmarks."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol, TypeVar

from vulkanocr.catalog import models_for
from vulkanocr.engine import OcrEngine

Item = TypeVar("Item")


class CropSource(Protocol):
    def crops(self, rgb: Any) -> list: ...


@contextmanager
def benchmark_engine(model_set: str, **options: Any) -> Iterator[OcrEngine]:
    """Build one engine and release it on success, refusal, or failure."""
    with OcrEngine(models_for(model_set), **options) as engine:
        yield engine


def require_nonempty(items: Sequence[Item], label: str) -> Sequence[Item]:
    """Return benchmark input or stop cleanly when measurement is undefined."""
    if not items:
        raise SystemExit(f"benchmark refused: no {label}; use an image containing text")
    return items


def require_text_crops(engine: CropSource, rgb: Any) -> list:
    return list(require_nonempty(engine.crops(rgb), "text crops detected"))


__all__ = ["benchmark_engine", "require_nonempty", "require_text_crops"]
