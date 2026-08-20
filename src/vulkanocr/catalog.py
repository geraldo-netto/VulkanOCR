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

import os
from dataclasses import dataclass
from pathlib import Path

from .engine import OcrModels

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
        _PACKAGED_KEYS,
        ("in0", "out0"),
        False,
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
    return Path(__file__).resolve().parents[2]


def _dictionary_path(spec_dictionary: str, base: Path) -> Path:
    """A dictionary shipped with this package, or one beside the models."""
    if spec_dictionary == _PACKAGED_KEYS:
        return Path(__file__).resolve().parent / "data" / _PACKAGED_KEYS
    return base / spec_dictionary


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
        dictionary=_dictionary_path(spec.dictionary, base),
        blobs=spec.blobs,
        dictionary_includes_blank=spec.dictionary_includes_blank,
    )
