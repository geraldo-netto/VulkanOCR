"""Explicit ncnn execution policy, independent from model and device choice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Precision = Literal["fp32", "fp16", "int8"]
PRECISIONS: tuple[Precision, ...] = ("fp32", "fp16", "int8")


@dataclass(frozen=True, slots=True)
class InferenceOptions:
    """Precision and backend switches applied to every ncnn network.

    Defaults preserve VulkanOCR's measured fp32 policy while stating ncnn's
    int8 defaults explicitly. Float models ignore enabled int8 inference,
    packing, and storage; quantized graphs require them.
    """

    use_vulkan_compute: bool = True
    use_fp16_packed: bool = False
    use_fp16_storage: bool = False
    use_fp16_arithmetic: bool = False
    use_int8_inference: bool = True
    use_int8_packed: bool = True
    use_int8_storage: bool = True
    use_int8_arithmetic: bool = False

    @classmethod
    def fp16(cls) -> InferenceOptions:
        """Vulkan execution with every fp16 optimization enabled."""
        return cls(
            use_fp16_packed=True,
            use_fp16_storage=True,
            use_fp16_arithmetic=True,
        )

    @classmethod
    def int8(cls) -> InferenceOptions:
        """Vulkan execution with quantized arithmetic enabled explicitly."""
        return cls(use_int8_arithmetic=True)


def options_for_precision(precision: str) -> InferenceOptions:
    """Execution options for one user-facing precision name."""
    if precision == "fp16":
        return InferenceOptions.fp16()
    if precision == "int8":
        return InferenceOptions.int8()
    if precision == "fp32":
        return InferenceOptions()
    raise ValueError(f"unknown precision {precision!r}; expected one of {', '.join(PRECISIONS)}")


__all__ = ["PRECISIONS", "InferenceOptions", "Precision", "options_for_precision"]
