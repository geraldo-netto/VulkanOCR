"""Explicit ncnn execution policy, independent from model and device choice."""

from __future__ import annotations

from dataclasses import dataclass


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


__all__ = ["InferenceOptions"]
