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

from .catalog import (
    CATALOG,
    DEFAULT_MODEL,
    DETECTORS,
    RECOGNIZERS,
    DetectorSpec,
    ModelProfile,
    RecognizerSpec,
    models_for,
    models_for_port,
)
from .device import HardwareVulkanUnavailableError, select_hardware_device
from .engine import OcrEngine, OcrEngineError, OcrLine, OcrModels, OcrResult
from .options import PRECISIONS, InferenceOptions, Precision, options_for_precision
from .policy import FalsePositivePolicy, RecognitionContext

__all__ = [
    "CATALOG",
    "DEFAULT_MODEL",
    "DETECTORS",
    "DetectorSpec",
    "FalsePositivePolicy",
    "HardwareVulkanUnavailableError",
    "InferenceOptions",
    "ModelProfile",
    "OcrEngine",
    "OcrEngineError",
    "OcrLine",
    "OcrModels",
    "OcrResult",
    "PRECISIONS",
    "Precision",
    "RECOGNIZERS",
    "RecognitionContext",
    "RecognizerSpec",
    "models_for",
    "models_for_port",
    "options_for_precision",
    "select_hardware_device",
]
