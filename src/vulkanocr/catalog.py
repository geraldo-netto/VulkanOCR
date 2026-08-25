"""Known ncnn model sets, as data.

Every component owns facts the engine cannot infer: where its files live,
what its tensors are called, whether a recognizer dictionary carries the CTC
blank, and any hard precision requirement. Recording them here keeps the
engine free of per-model branches — adding a port is data, not new code.

Provenance:

* ``v5-*``    nihui/ncnn-android-ppocrv5, BSD 3-Clause, Tencent, 2026-05-27.
* ``v6-*``    Avafly/PaddleOCR-ncnn-CPP release v0.3.0, MIT, 2026-06-13.

Both are third-party ncnn conversions of PaddlePaddle's Apache-2.0 PP-OCR
weights; neither upstream is modified.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .engine import OcrModels
from .options import Precision

# PP-OCRv6 medium is the default: it is the current PaddleOCR generation and,
# measured on this host's RX 6600 XT against the same page, it was the only
# tier that produced no low-confidence noise lines.
# The nihui port ships no dictionary, so the class list extracted from its
# header lives with this package rather than with the downloaded models.
_PACKAGED_KEYS = "ppocrv5_keys.txt"
DEFAULT_MODEL = "v6-medium"

_NIHUI = "nihui-port/app/src/main/assets"
_AVAFLY = "PaddleOCR-ncnn-CPP/models"


@dataclass(frozen=True, slots=True)
class DetectorSpec:
    """One detection graph and the runtime facts intrinsic to it.

    ``required_precision`` is ``None`` for ordinary float graphs. A graph
    converted specifically for quantized inference states ``"int8"`` here,
    independently from whichever recognizer a profile pairs it with.
    """

    param: str
    blobs: tuple[str, str]
    required_precision: Precision | None = None


@dataclass(frozen=True, slots=True)
class RecognizerSpec:
    """One recognition graph, its character map, and its runtime facts."""

    param: str
    dictionary: str
    blobs: tuple[str, str]
    dictionary_includes_blank: bool
    required_precision: Precision | None = None

    @property
    def ctc_offset(self) -> int:
        """How this recognizer's class indices address its dictionary."""
        return 0 if self.dictionary_includes_blank else 1


# Component names are stable seams for future profiles. A language-specific
# or quantized recognizer can be added without copying a detector record.
DETECTORS: dict[str, DetectorSpec] = {
    "v6-medium-det": DetectorSpec(
        f"{_AVAFLY}/PP_OCRv6_medium_det.param",
        ("input", "output"),
    ),
    "v6-small-det": DetectorSpec(
        f"{_AVAFLY}/PP_OCRv6_small_det.param",
        ("input", "output"),
    ),
    "v6-tiny-det": DetectorSpec(
        f"{_AVAFLY}/PP_OCRv6_tiny_det.param",
        ("input", "output"),
    ),
    "v5-mobile-det": DetectorSpec(
        f"{_NIHUI}/PP_OCRv5_mobile_det.ncnn.param",
        ("in0", "out0"),
    ),
}

RECOGNIZERS: dict[str, RecognizerSpec] = {
    "v6-medium-rec": RecognizerSpec(
        f"{_AVAFLY}/PP_OCRv6_medium_rec.param",
        f"{_AVAFLY}/ppocr_keys_v6.txt",
        ("input", "output"),
        True,
    ),
    "v6-small-rec": RecognizerSpec(
        f"{_AVAFLY}/PP_OCRv6_small_rec.param",
        f"{_AVAFLY}/ppocr_keys_v6.txt",
        ("input", "output"),
        True,
    ),
    "v6-tiny-rec": RecognizerSpec(
        f"{_AVAFLY}/PP_OCRv6_tiny_rec.param",
        f"{_AVAFLY}/ppocr_keys_v6_tiny.txt",
        ("input", "output"),
        True,
    ),
    "v5-mobile-rec": RecognizerSpec(
        f"{_NIHUI}/PP_OCRv5_mobile_rec.ncnn.param",
        _PACKAGED_KEYS,
        ("in0", "out0"),
        False,
    ),
}


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """A named pairing of independently reusable inference components."""

    detector: DetectorSpec
    recognizer: RecognizerSpec
    note: str
    orientation: str | None = None
    orientation_blobs: tuple[str, str] = ("input", "output")
    orientation_labels: tuple[int, int] = (0, 180)

    @property
    def precision_requirements(self) -> frozenset[Precision]:
        """Hard requirements contributed by either component."""
        return frozenset(
            required
            for required in (
                self.detector.required_precision,
                self.recognizer.required_precision,
            )
            if required is not None
        )


CATALOG: dict[str, ModelProfile] = {
    "v6-medium": ModelProfile(
        DETECTORS["v6-medium-det"],
        RECOGNIZERS["v6-medium-rec"],
        "current generation, highest-accuracy catalog tier",
        f"{_AVAFLY}/PP_LCNet_x0_25_textline_ori.param",
    ),
    "v6-small": ModelProfile(
        DETECTORS["v6-small-det"],
        RECOGNIZERS["v6-small-rec"],
        "balanced tier",
        f"{_AVAFLY}/PP_LCNet_x0_25_textline_ori.param",
    ),
    "v6-tiny": ModelProfile(
        DETECTORS["v6-tiny-det"],
        RECOGNIZERS["v6-tiny-rec"],
        "fastest catalog tier, 49 languages",
        f"{_AVAFLY}/PP_LCNet_x0_25_textline_ori.param",
    ),
    "v5-mobile": ModelProfile(
        DETECTORS["v5-mobile-det"],
        RECOGNIZERS["v5-mobile-rec"],
        "previous generation, kept for comparison",
    ),
}


MODELS_ROOT_VARIABLE = "VULKANOCR_MODELS_ROOT"


def default_models_root() -> Path:
    """Where the downloaded model ports live.

    The graphs are third-party ports of PaddleOCR's weights and are not
    shipped with this package, so their location is somebody's choice rather
    than a fact about the install: `VULKANOCR_MODELS_ROOT` names it, and the
    fallback is the checkout this file sits in, which is where the setup in
    the README clones them.
    """
    configured = os.environ.get(MODELS_ROOT_VARIABLE, "").strip()
    if configured:
        return Path(configured).expanduser()
    # From a checkout, parents[2] is the repository root, where the README's
    # setup clones the model repositories. From an installed wheel the same
    # hop lands in site-packages' parent — nothing lives there, so the miss
    # is reported against a root that at least exists and is settable.
    return Path(__file__).resolve().parents[2]


def _dictionary_path(spec_dictionary: str, base: Path) -> Path:
    """A dictionary shipped with this package, or one beside the models."""
    if spec_dictionary == _PACKAGED_KEYS:
        return Path(__file__).resolve().parent / "data" / _PACKAGED_KEYS
    return base / spec_dictionary


# The per-port components an external consumer needs when it holds model
# files outside this catalog's clone layout (VOCR-0061). Naming the port
# inherits its blob, blank, and precision facts without copying paths.
PORT_FACTS: dict[str, tuple[DetectorSpec, RecognizerSpec]] = {
    "avafly-v6": (DETECTORS["v6-medium-det"], RECOGNIZERS["v6-medium-rec"]),
    "nihui-v5": (DETECTORS["v5-mobile-det"], RECOGNIZERS["v5-mobile-rec"]),
}


def models_for_port(
    port: str,
    det_param: Path,
    rec_param: Path,
    dictionary: Path,
) -> OcrModels:
    """An :class:`OcrModels` from explicit paths and a named port's facts."""
    try:
        detector, recognizer = PORT_FACTS[port]
    except KeyError:
        known = ", ".join(sorted(PORT_FACTS))
        raise ValueError(f"unknown model port {port!r}; known ports: {known}") from None
    return OcrModels(
        det_param=Path(det_param),
        rec_param=Path(rec_param),
        dictionary=Path(dictionary),
        blobs=detector.blobs,
        recognizer_blobs=recognizer.blobs,
        dictionary_includes_blank=recognizer.dictionary_includes_blank,
        detector_required_precision=detector.required_precision,
        recognizer_required_precision=recognizer.required_precision,
    )


def models_for(name: str = DEFAULT_MODEL, root: Path | None = None) -> OcrModels:
    """Build an :class:`OcrModels` record for a catalogued model set."""
    try:
        profile = CATALOG[name]
    except KeyError:
        known = ", ".join(sorted(CATALOG))
        raise ValueError(f"unknown model set {name!r}; known sets: {known}") from None
    base = Path(root) if root is not None else default_models_root()
    detector = profile.detector
    recognizer = profile.recognizer
    return OcrModels(
        det_param=base / detector.param,
        rec_param=base / recognizer.param,
        dictionary=_dictionary_path(recognizer.dictionary, base),
        blobs=detector.blobs,
        recognizer_blobs=recognizer.blobs,
        dictionary_includes_blank=recognizer.dictionary_includes_blank,
        orientation_param=(base / profile.orientation if profile.orientation is not None else None),
        orientation_blobs=profile.orientation_blobs,
        orientation_labels=profile.orientation_labels,
        detector_required_precision=detector.required_precision,
        recognizer_required_precision=recognizer.required_precision,
    )
