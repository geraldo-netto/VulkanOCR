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
from .options import InferenceOptions, Precision
from .orientation import classify_patch_orientation, rotate_patch
from .recognition import CtcDictionaryMismatchError, crop_region, decode_ctc, patch_logits

DEFAULT_TARGET_SIZE = 640


class OcrEngineError(RuntimeError):
    """A stable engine failure with a code the caller can branch on."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class OcrModels:
    """Resolved model paths and component-specific runtime facts.

    Ports differ in tensor naming: the nihui PP-OCRv5 graphs use ``in0``/``out0``,
    the Avafly PP-OCRv6 graphs use ``input``/``output``. Naming is data, not a
    code branch, so a new port is a new record rather than a new engine. ``blobs``
    remains the detector and legacy shared convention; ``recognizer_blobs``
    overrides it when independently composed components use different names.
    """

    det_param: Path
    rec_param: Path
    dictionary: Path
    blobs: tuple[str, str] = ("in0", "out0")
    dictionary_includes_blank: bool = False
    orientation_param: Path | None = None
    orientation_blobs: tuple[str, str] = ("input", "output")
    orientation_labels: tuple[int, int] = (0, 180)
    recognizer_blobs: tuple[str, str] | None = None
    detector_required_precision: Precision | None = None
    recognizer_required_precision: Precision | None = None

    @property
    def detector_blobs(self) -> tuple[str, str]:
        """Blob names owned by the selected detector."""
        return self.blobs

    @property
    def resolved_recognizer_blobs(self) -> tuple[str, str]:
        """Blob names owned by the selected recognizer."""
        return self.recognizer_blobs or self.blobs

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

    def validated(self, nets: tuple[str, ...] = ("det", "rec")) -> OcrModels:
        """These files exist on disk — the halves asked for, dictionary always.

        A recognition-only engine (VOCR-0042) used to be refused over a
        missing detection graph it would never load (VOCR-0052): a half
        engine's contract validates half the catalog entry.
        """
        required: list[Path] = [self.dictionary]
        if "det" in nets:
            required += [self.det_param, self.det_param.with_suffix(".bin")]
        if "rec" in nets:
            required += [self.rec_param, self.rec_param.with_suffix(".bin")]
        if "ori" in nets:
            if self.orientation_param is None:
                raise OcrEngineError(
                    "model-missing", "the selected model profile has no orientation graph"
                )
            required += [
                self.orientation_param,
                self.orientation_param.with_suffix(".bin"),
            ]
        for path in required:
            if not Path(path).is_file():
                raise OcrEngineError(
                    "model-missing",
                    f"model file is absent: {path}\n"
                    "The model graphs are third-party ncnn ports and are not shipped "
                    "with this package. Clone them where THIRD-PARTY.md describes — "
                    "github.com/Avafly/PaddleOCR-ncnn-CPP for PP-OCRv6, "
                    "github.com/nihui/ncnn-android-ppocrv5 for PP-OCRv5 — and point "
                    "VULKANOCR_MODELS_ROOT at the directory that contains them.",
                )
        return self


@dataclass(frozen=True, slots=True)
class OcrLine:
    """One recognised text line in original-image coordinates.

    The box is an oriented rectangle, so its sides are named for what they
    are: ``thickness`` is the stroke-to-stroke height of the text and
    ``length`` its reading-direction extent. The fields were called
    ``width``/``height`` once, and on a horizontal line reported width 54 and
    height 356 — every consumer that drew an axis-aligned box from them got
    it rotated a quarter turn. ``bounding_box()`` is the axis-aligned answer.
    """

    text: str
    confidence: float
    box_score: float
    center_x: float
    center_y: float
    thickness: float
    length: float
    angle: float
    vertical: bool

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Axis-aligned ``(left, top, right, bottom)`` enclosing the line."""
        import cv2  # noqa: PLC0415 - only needed when somebody asks for a box
        import numpy as np  # noqa: PLC0415

        corners = cv2.boxPoints(
            ((self.center_x, self.center_y), (self.thickness, self.length), self.angle)
        )
        xs, ys = np.asarray(corners)[:, 0], np.asarray(corners)[:, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


@dataclass(frozen=True, slots=True)
class OcrResult:
    """What one read produced, and what it declined to guess at.

    ``lines`` holds only regions whose decode produced text; a detected box
    that decoded to nothing is counted in ``undecoded_regions`` rather than
    silently vanishing, so a caller can tell "clean page" from "the detector
    saw something the recogniser could not read" — which is exactly how the
    30°–60° rotation bug stayed invisible for as long as it did.
    """

    device_name: str
    lines: tuple[OcrLine, ...]
    undecoded_regions: int = 0


class OcrEngine:
    """Detection + recognition on one hardware Vulkan device, loaded once."""

    def __init__(
        self,
        models: OcrModels,
        *,
        runtime: Any = None,
        target_size: int = DEFAULT_TARGET_SIZE,
        use_vulkan: bool | None = None,
        use_fp16: bool | None = None,
        options: InferenceOptions | None = None,
        device: Any = None,
        nets: tuple[str, ...] | None = None,
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
        # The nets choice is checked before it decides what must exist.
        nets = self._resolved_nets(models, nets)
        self._models = models.validated(nets)
        self._target_size = self._validated_target_size(target_size)
        self._options = self._resolved_options(options, use_vulkan, use_fp16)
        # A caller may pin a specific device — the parallel engine builds one
        # engine per card — otherwise the most capable one is selected.
        self._device = device if device is not None else select_hardware_device(runtime)
        self._characters = self._load_dictionary(models.dictionary)
        # A caller may ask for half an engine (VOCR-0042): a pool worker only
        # ever recognises, and its unused detection net still held hundreds
        # of MiB of Vulkan allocations on every device.
        self._det = None
        self._ori = None
        self._rec = None
        try:
            if "det" in nets:
                self._det = self._load_net(models.det_param)
            if "ori" in nets:
                if models.orientation_param is None:  # guarded by validated(), narrows the type
                    raise OcrEngineError("model-missing", "orientation graph is absent")
                self._ori = self._load_net(models.orientation_param)
            if "rec" in nets:
                self._rec = self._load_net(models.rec_param)
        except BaseException:
            # A recognition graph that refuses to load must not strand the
            # detection net's ~hundreds of MiB of Vulkan allocations on an
            # object the caller never receives.
            self.close()
            raise

    @property
    def device_name(self) -> str:
        return self._device.name

    def close(self) -> None:
        """Release every loaded net and its Vulkan allocations.

        An engine holds on the order of 700 MiB of VRAM (measured in
        docs/benchmarks.md), and nothing freed it before: a benchmark that
        built engines in a loop accumulated one device's worth per pass.
        Safe to call twice, and called for you by the context manager.
        """
        for name in ("_det", "_ori", "_rec"):
            net = getattr(self, name, None)
            if net is not None:
                net.clear()
                setattr(self, name, None)

    def __enter__(self) -> OcrEngine:
        return self

    def __exit__(self, *_exception) -> None:
        self.close()

    def detect(self, rgb: np.ndarray) -> list:
        """The text regions on one validated RGB image, unrecognised.

        Public because the benchmarks are consumers too: they time the two
        halves apart, and while this seam was private they reached for
        ``engine._det`` and ``engine._runtime`` sixty-five times — code no
        checker could protect from a rename.
        """
        self._validated(rgb)
        if self._det is None:
            raise OcrEngineError("net-unloaded", "this engine was built without the detection net")
        return detect_regions(
            self._runtime,
            self._det,
            rgb,
            self._target_size,
            self._models.detector_blobs,
        )

    def crops(self, rgb: np.ndarray) -> list[tuple]:
        """Every detected region with its rectified 48-high patch, empties dropped."""
        pairs = []
        for region in self.detect(rgb):
            patch = crop_region(rgb, region)
            if patch.size:
                if self._ori is not None:
                    patch = self.orient(patch)
                pairs.append((region, patch))
        return pairs

    def orient(self, patch: np.ndarray) -> np.ndarray:
        """Rotate a rectified line to the recognizer's expected direction."""
        self._validated_patch(patch)
        if self._ori is None:
            raise OcrEngineError(
                "net-unloaded", "this engine was built without the orientation net"
            )
        degrees, _confidence = classify_patch_orientation(
            self._runtime,
            self._ori,
            patch,
            self._models.orientation_blobs,
            self._models.orientation_labels,
        )
        return rotate_patch(patch, degrees)

    def recognise(self, patch: np.ndarray) -> tuple[str, float]:
        """One rectified patch through the recognition net and the CTC decode."""
        return self.decode(self.logits(patch))

    def logits(self, patch: np.ndarray) -> np.ndarray:
        """One patch's raw CTC logits, for callers that decode segments themselves."""
        self._validated_patch(patch)
        if self._rec is None:
            raise OcrEngineError(
                "net-unloaded", "this engine was built without the recognition net"
            )
        return patch_logits(
            self._runtime,
            self._rec,
            patch,
            self._models.resolved_recognizer_blobs,
        )

    def decode(self, logits: np.ndarray) -> tuple[str, float]:
        """Greedy-decode a logits slice with this engine's dictionary and offset."""
        try:
            return decode_ctc(logits, self._characters, self._models.ctc_offset)
        except CtcDictionaryMismatchError as error:
            raise OcrEngineError("dictionary-mismatch", str(error)) from error

    @staticmethod
    def _resolved_nets(models: OcrModels, nets: tuple[str, ...] | None) -> tuple[str, ...]:
        if nets is None:
            nets = ("det", "ori", "rec") if models.orientation_param is not None else ("det", "rec")
        return OcrEngine._validated_nets(nets)

    @staticmethod
    def _validated_nets(nets: tuple[str, ...]) -> tuple[str, ...]:
        if not nets or any(name not in ("det", "ori", "rec") for name in nets):
            raise OcrEngineError("nets-invalid", "nets must contain only 'det', 'ori', and 'rec'")
        return nets

    @staticmethod
    def _validated(rgb: np.ndarray) -> None:
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise OcrEngineError("image-invalid", "expected an RGB HxWx3 array")
        if rgb.dtype != np.uint8:
            raise OcrEngineError("image-invalid", "expected uint8 pixels")
        if rgb.shape[0] < 1 or rgb.shape[1] < 1:
            raise OcrEngineError("image-invalid", "expected non-empty image dimensions")

    @staticmethod
    def _validated_patch(patch: np.ndarray) -> None:
        if (
            not isinstance(patch, np.ndarray)
            or patch.ndim != 3
            or patch.shape[2] != 3
            or patch.shape[0] != 48
        ):
            raise OcrEngineError("patch-invalid", "expected an RGB 48xWx3 recognition patch")
        if patch.dtype != np.uint8:
            raise OcrEngineError("patch-invalid", "expected uint8 patch pixels")
        if patch.shape[1] < 1:
            raise OcrEngineError("patch-invalid", "expected a non-empty recognition patch")

    @staticmethod
    def _validated_target_size(target_size: Any) -> int:
        try:
            value = int(target_size)
        except (TypeError, ValueError) as error:
            raise OcrEngineError(
                "target-size-invalid", "target_size must be a positive integer"
            ) from error
        if value <= 0:
            raise OcrEngineError("target-size-invalid", "target_size must be a positive integer")
        return value

    def read(self, rgb: np.ndarray) -> OcrResult:
        """Recognise every text line in an RGB uint8 array."""
        return assemble_result(
            self._device.name,
            ((region, self.recognise(patch)) for region, patch in self.crops(rgb)),
        )

    def _load_net(self, param: Path):
        net = self._runtime.Net()
        loaded = False
        for name in InferenceOptions.__dataclass_fields__:
            setattr(net.opt, name, getattr(self._options, name))
        if self._options.use_vulkan_compute:
            net.set_vulkan_device(self._device.index)
        try:
            if net.load_param(str(param)) != 0:
                raise OcrEngineError("model-invalid", f"cannot parse ncnn param: {param}")
            if net.load_model(str(param.with_suffix(".bin"))) != 0:
                raise OcrEngineError("model-invalid", f"cannot load ncnn weights for: {param}")
            loaded = True
        finally:
            if not loaded:
                net.clear()
        return net

    @staticmethod
    def _resolved_options(
        options: InferenceOptions | None,
        use_vulkan: bool | None,
        use_fp16: bool | None,
    ) -> InferenceOptions:
        if options is not None:
            if use_vulkan is not None or use_fp16 is not None:
                raise OcrEngineError(
                    "options-conflict",
                    "options cannot be combined with use_vulkan or use_fp16",
                )
            return options
        fp16 = bool(use_fp16) if use_fp16 is not None else False
        return InferenceOptions(
            use_vulkan_compute=bool(use_vulkan) if use_vulkan is not None else True,
            use_fp16_packed=fp16,
            use_fp16_storage=fp16,
            use_fp16_arithmetic=fp16,
        )

    @staticmethod
    def _load_dictionary(path: Path) -> tuple[str, ...]:
        """One class per line — including the final literal-space class.

        Not ``splitlines()`` and not ``strip()``: every keys file ends with a
        line holding a single ASCII space, and both would eat it, shifting
        every decoded character by one. A trailing newline, though, is a file
        convention rather than a class — the Avafly clone's ``ppocr_keys_v5``
        has one and the packaged copy does not — so exactly one final empty
        entry is dropped when present.
        """
        characters = Path(path).read_text(encoding="utf-8").split("\n")
        if characters and characters[-1] == "":
            characters.pop()
        if len(characters) < 2:
            raise OcrEngineError("dictionary-invalid", "dictionary has fewer than two classes")
        return tuple(characters)


def assemble_result(device_name: str, recognised) -> OcrResult:
    """One `OcrResult` from `(region, (text, confidence))` pairs.

    The single assembly both engines share: `OcrEngine.read` feeds it
    straight off its own recognitions and `ParallelOcr` from replies that
    arrive out of order. It existed twice, line for line, and a change to
    the sort key or a new `OcrLine` field in one silently diverged the
    other (VOCR-0041).
    """
    lines = []
    undecoded = 0
    for region, (text, confidence) in recognised:
        if not text:
            undecoded += 1
            continue
        lines.append(
            OcrLine(
                text=text,
                confidence=confidence,
                box_score=region.score,
                center_x=region.center_x,
                center_y=region.center_y,
                thickness=region.width,
                length=region.height,
                angle=region.angle,
                vertical=region.vertical,
            )
        )
    lines.sort(key=lambda line: (line.center_y, line.center_x))
    return OcrResult(device_name=device_name, lines=tuple(lines), undecoded_regions=undecoded)
