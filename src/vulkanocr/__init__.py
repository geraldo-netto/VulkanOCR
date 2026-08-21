"""PaddleOCR's models on ncnn over Vulkan: GPU OCR without PaddlePaddle.

Dependencies are ncnn, numpy and OpenCV only, and both upstreams are
unmodified externals: the models are third-party ncnn ports of PaddleOCR's
weights — PP-OCRv6 tiny/small/medium and PP-OCRv5 mobile, catalogued in
:mod:`vulkanocr.catalog` — and the dictionary is a plain text file, one
class per line.

Public surface:

    with OcrEngine(models_for("v6-medium")) as engine:
        result = engine.read(rgb_array)   # -> OcrResult(lines=[OcrLine, ...])

The engine also offers its halves — detect / crops / recognise / logits /
decode — for callers that time or compose them separately.
"""

from .catalog import CATALOG, DEFAULT_MODEL, models_for, models_for_port
from .device import HardwareVulkanUnavailableError, select_hardware_device
from .engine import OcrEngine, OcrEngineError, OcrLine, OcrModels, OcrResult

__all__ = [
    "CATALOG",
    "DEFAULT_MODEL",
    "HardwareVulkanUnavailableError",
    "OcrEngine",
    "OcrEngineError",
    "OcrLine",
    "OcrModels",
    "OcrResult",
    "models_for",
    "models_for_port",
    "select_hardware_device",
]
