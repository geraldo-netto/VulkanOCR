"""The engine facade: load once, read many images, Vulkan only.

Precision policy matches the service's Vulkan executor: fp16 packed, storage,
and arithmetic are all disabled, so every model runs at fp32 regardless of the
device's fp16 capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .detection import detect_regions
from .device import select_hardware_device
from .recognition import crop_region, decode_ctc, patch_logits, recognise_patch

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

    @property
    def ctc_offset(self) -> int:
        """How class indices map to the dictionary: 0 when it carries the blank.

        Getting this wrong does not crash — it shifts every character by one
        and produces fluent-looking nonsense — and the ternary that derives it
        used to be written out at eight call sites, each an independent chance
        to invert it. It is a fact about the port, so it lives with the rest
        of the port's facts.
        """
        return 0 if self.dictionary_includes_blank else 1

    def validated(self) -> OcrModels:
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

    def __init__(
        self,
        models: OcrModels,
        *,
        runtime: Any = None,
        target_size: int = DEFAULT_TARGET_SIZE,
    ):
        """`runtime` is the ncnn module, or anything shaped like it.

        Typed as `Any` on purpose: it is a seam, not a dependency. The tests
        pass a fake with the three attributes this class touches, and the ncnn
        wheel ships no stubs, so a narrower annotation would describe the
        installed package rather than what this class requires.
        """
        if runtime is None:
            try:
                import ncnn as runtime  # type: ignore[no-redef]  # noqa: PLC0415
            except ImportError as error:  # pragma: no cover - environment boundary
                raise OcrEngineError("runtime-missing", "ncnn is not installed") from error
        self._runtime: Any = runtime
        self._models = models.validated()
        self._target_size = int(target_size)
        self._device = select_hardware_device(runtime)
        self._characters = self._load_dictionary(models.dictionary)
        self._det = self._load_net(models.det_param)
        self._rec = self._load_net(models.rec_param)

    @property
    def device_name(self) -> str:
        return self._device.name

    def detect(self, rgb: np.ndarray) -> list:
        """The text regions on one validated RGB image, unrecognised.

        Public because the benchmarks are consumers too: they time the two
        halves apart, and while this seam was private they reached for
        ``engine._det`` and ``engine._runtime`` sixty-five times — code no
        checker could protect from a rename.
        """
        self._validated(rgb)
        return detect_regions(self._runtime, self._det, rgb, self._target_size, self._models.blobs)

    def crops(self, rgb: np.ndarray) -> list[tuple]:
        """Every detected region with its rectified 48-high patch, empties dropped."""
        pairs = []
        for region in self.detect(rgb):
            patch = crop_region(rgb, region)
            if patch.size:
                pairs.append((region, patch))
        return pairs

    def recognise(self, patch: np.ndarray) -> tuple[str, float]:
        """One rectified patch through the recognition net and the CTC decode."""
        return recognise_patch(
            self._runtime,
            self._rec,
            patch,
            self._characters,
            self._models.blobs,
            self._models.ctc_offset,
        )

    def logits(self, patch: np.ndarray) -> np.ndarray:
        """One patch's raw CTC logits, for callers that decode segments themselves."""
        return patch_logits(self._runtime, self._rec, patch, self._models.blobs)

    def decode(self, logits: np.ndarray) -> tuple[str, float]:
        """Greedy-decode a logits slice with this engine's dictionary and offset."""
        return decode_ctc(logits, self._characters, self._models.ctc_offset)

    @staticmethod
    def _validated(rgb: np.ndarray) -> None:
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise OcrEngineError("image-invalid", "expected an RGB HxWx3 array")
        if rgb.dtype != np.uint8:
            raise OcrEngineError("image-invalid", "expected uint8 pixels")
        if rgb.shape[0] * rgb.shape[1] > MAX_IMAGE_PIXELS:
            raise OcrEngineError("image-too-large", "image exceeds the pixel bound")

    def read(self, rgb: np.ndarray) -> OcrResult:
        """Recognise every text line in an RGB uint8 array."""
        lines = []
        for region, patch in self.crops(rgb):
            text, confidence = self.recognise(patch)
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
