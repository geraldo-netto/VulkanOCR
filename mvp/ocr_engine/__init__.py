"""Decoupled PaddleOCR-on-ncnn engine running on Vulkan.

This package is the OMNI-0351 MVP: it depends on ncnn, numpy, and OpenCV
only — no PaddlePaddle, no PaddleOCR, no omnitensor imports — and it treats
both upstreams as unmodified externals. Models are PP-OCRv5 graphs already
ported to ncnn form; the dictionary is a plain text file, one class per line.

Public surface:

    engine = OcrEngine(OcrModels(det_param, rec_param, dictionary))
    result = engine.read(rgb_array)   # -> OcrResult(lines=[OcrLine, ...])
"""

from .catalog import CATALOG, DEFAULT_MODEL, models_for
from .device import HardwareVulkanUnavailable, select_hardware_device
from .engine import OcrEngine, OcrEngineError, OcrLine, OcrModels, OcrResult

__all__ = [
    "CATALOG",
    "DEFAULT_MODEL",
    "HardwareVulkanUnavailable",
    "OcrEngine",
    "OcrEngineError",
    "OcrLine",
    "OcrModels",
    "OcrResult",
    "models_for",
    "select_hardware_device",
]
