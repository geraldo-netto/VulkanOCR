"""The engine facade: load once, read many images, Vulkan only.

Precision policy matches the service's Vulkan executor: fp16 packed, storage,
and arithmetic are all disabled, so every model runs at fp32 regardless of the
device's fp16 capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detection import detect_regions
from .device import select_hardware_device
from .recognition import crop_region, recognise_patch

DEFAULT_TARGET_SIZE = 640
MAX_IMAGE_PIXELS = 64_000_000


class OcrEngineError(RuntimeError):
    """A stable engine failure with a code the caller can branch on."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class OcrModels:
    """Paths to the ncnn model pair, the class dictionary, and the blob names.

    Ports differ in tensor naming: the nihui PP-OCRv5 graphs use ``in0``/``out0``,
    the Avafly PP-OCRv6 graphs use ``input``/``output``. Naming is data, not a
    code branch, so a new port is a new record rather than a new engine.
    """

    det_param: Path
    rec_param: Path
    dictionary: Path
    blobs: tuple = ("in0", "out0")
    dictionary_includes_blank: bool = False

    def validated(self) -> "OcrModels":
        for path in (
            self.det_param,
            self.det_param.with_suffix(".bin"),
            self.rec_param,
            self.rec_param.with_suffix(".bin"),
            self.dictionary,
        ):
            if not Path(path).is_file():
                raise OcrEngineError("model-missing", f"model file is absent: {path}")
        return self


@dataclass(frozen=True, slots=True)
class OcrLine:
    """One recognised text line in original-image coordinates."""

    text: str
    confidence: float
    box_score: float
    center_x: float
    center_y: float
    width: float
    height: float
    angle: float
    vertical: bool


@dataclass(frozen=True, slots=True)
class OcrResult:
    device_name: str
    lines: tuple[OcrLine, ...]


class OcrEngine:
    """Detection + recognition on one hardware Vulkan device, loaded once."""

    def __init__(self, models: OcrModels, *, runtime=None, target_size: int = DEFAULT_TARGET_SIZE):
        if runtime is None:
            try:
                import ncnn as runtime  # type: ignore[no-redef]  # noqa: PLC0415
            except ImportError as error:  # pragma: no cover - environment boundary
                raise OcrEngineError("runtime-missing", "ncnn is not installed") from error
        self._runtime = runtime
        self._models = models.validated()
        self._target_size = int(target_size)
        self._device = select_hardware_device(runtime)
        self._characters = self._load_dictionary(models.dictionary)
        self._det = self._load_net(models.det_param)
        self._rec = self._load_net(models.rec_param)

    @property
    def device_name(self) -> str:
        return self._device.name

    def read(self, rgb: np.ndarray) -> OcrResult:
        """Recognise every text line in an RGB uint8 array."""
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise OcrEngineError("image-invalid", "expected an RGB HxWx3 array")
        if rgb.dtype != np.uint8:
            raise OcrEngineError("image-invalid", "expected uint8 pixels")
        if rgb.shape[0] * rgb.shape[1] > MAX_IMAGE_PIXELS:
            raise OcrEngineError("image-too-large", "image exceeds the pixel bound")

        regions = detect_regions(
            self._runtime, self._det, rgb, self._target_size, self._models.blobs
        )
        lines = []
        for region in regions:
            patch = crop_region(rgb, region)
            if patch.size == 0:
                continue
            text, confidence = recognise_patch(
                self._runtime,
                self._rec,
                patch,
                self._characters,
                self._models.blobs,
                0 if self._models.dictionary_includes_blank else 1,
            )
            if not text:
                continue
            lines.append(
                OcrLine(
                    text=text,
                    confidence=confidence,
                    box_score=region.score,
                    center_x=region.center_x,
                    center_y=region.center_y,
                    width=region.width,
                    height=region.height,
                    angle=region.angle,
                    vertical=region.vertical,
                )
            )
        lines.sort(key=lambda line: (line.center_y, line.center_x))
        return OcrResult(device_name=self._device.name, lines=tuple(lines))

    def _load_net(self, param: Path):
        net = self._runtime.Net()
        net.opt.use_vulkan_compute = True
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        net.set_vulkan_device(self._device.index)
        if net.load_param(str(param)) != 0:
            raise OcrEngineError("model-invalid", f"cannot parse ncnn param: {param}")
        if net.load_model(str(param.with_suffix(".bin"))) != 0:
            raise OcrEngineError("model-invalid", f"cannot load ncnn weights for: {param}")
        return net

    @staticmethod
    def _load_dictionary(path: Path) -> tuple[str, ...]:
        characters = tuple(Path(path).read_text(encoding="utf-8").split("\n"))
        if len(characters) < 2:
            raise OcrEngineError("dictionary-invalid", "dictionary has fewer than two classes")
        return characters
