"""Known ncnn model sets, as data.

Every port differs in three facts the engine cannot infer: where its files
live, what its tensors are called, and whether its dictionary carries the CTC
blank as its own first entry. Recording them here keeps the engine free of
per-model branches — adding a port is a new record, not new code.

Provenance:

* ``v5-*``    nihui/ncnn-android-ppocrv5, BSD 3-Clause, Tencent, 2026-05-27.
* ``v6-*``    Avafly/PaddleOCR-ncnn-CPP release v0.3.0, MIT, 2026-06-13.

Both are third-party ncnn conversions of PaddlePaddle's Apache-2.0 PP-OCR
weights; neither upstream is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .engine import OcrModels

# PP-OCRv6 medium is the default: it is the current PaddleOCR generation and,
# measured on this host's RX 6600 XT against the same page, it was the only
# tier that produced no low-confidence noise lines.
DEFAULT_MODEL = "v6-medium"

_NIHUI = "nihui-port/app/src/main/assets"
_AVAFLY = "PaddleOCR-ncnn-CPP/models"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One installable model set, relative to a models root."""

    det: str
    rec: str
    dictionary: str
    blobs: tuple[str, str]
    dictionary_includes_blank: bool
    note: str


CATALOG: dict[str, ModelSpec] = {
    "v6-medium": ModelSpec(
        f"{_AVAFLY}/PP_OCRv6_medium_det.param",
        f"{_AVAFLY}/PP_OCRv6_medium_rec.param",
        f"{_AVAFLY}/ppocr_keys_v6.txt",
        ("input", "output"),
        True,
        "current generation, highest accuracy, ~1.0 s per page here",
    ),
    "v6-small": ModelSpec(
        f"{_AVAFLY}/PP_OCRv6_small_det.param",
        f"{_AVAFLY}/PP_OCRv6_small_rec.param",
        f"{_AVAFLY}/ppocr_keys_v6.txt",
        ("input", "output"),
        True,
        "balanced tier",
    ),
    "v6-tiny": ModelSpec(
        f"{_AVAFLY}/PP_OCRv6_tiny_det.param",
        f"{_AVAFLY}/PP_OCRv6_tiny_rec.param",
        f"{_AVAFLY}/ppocr_keys_v6_tiny.txt",
        ("input", "output"),
        True,
        "fastest tier, 49 languages, ~0.37 s per page here",
    ),
    "v5-mobile": ModelSpec(
        f"{_NIHUI}/PP_OCRv5_mobile_det.ncnn.param",
        f"{_NIHUI}/PP_OCRv5_mobile_rec.ncnn.param",
        "ppocrv5_keys.txt",
        ("in0", "out0"),
        False,
        "previous generation, kept for comparison",
    ),
}


def default_models_root() -> Path:
    """The spike checkout that holds the downloaded model sets."""
    return Path(__file__).resolve().parent.parent.parent


def models_for(name: str = DEFAULT_MODEL, root: Path | None = None) -> OcrModels:
    """Build an :class:`OcrModels` record for a catalogued model set."""
    try:
        spec = CATALOG[name]
    except KeyError:
        known = ", ".join(sorted(CATALOG))
        raise ValueError(f"unknown model set {name!r}; known sets: {known}") from None
    base = Path(root) if root is not None else default_models_root()
    return OcrModels(
        det_param=base / spec.det,
        rec_param=base / spec.rec,
        dictionary=base / spec.dictionary,
        blobs=spec.blobs,
        dictionary_includes_blank=spec.dictionary_includes_blank,
    )
