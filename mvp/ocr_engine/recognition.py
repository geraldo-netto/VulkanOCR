"""Text recognition: oriented crop, CTC head, greedy decode.

Constants follow nihui/ncnn-android-ppocrv5 ``ppocrv5.cpp`` (BSD 3-Clause,
Tencent). The dictionary is injected as a plain sequence: CTC class 0 is the
blank, class ``i`` maps to ``characters[i - 1]``.
"""

from __future__ import annotations

import cv2
import numpy as np

MEAN = (127.5, 127.5, 127.5)
NORM = (1 / 127.5, 1 / 127.5, 1 / 127.5)
TARGET_HEIGHT = 48


def crop_region(rgb: np.ndarray, region) -> np.ndarray:
    """Affine-rectify one oriented region to a horizontal 48-high patch."""
    target_width = max(int(region.height * TARGET_HEIGHT / max(region.width, 1e-6)), 1)
    corners = cv2.boxPoints(region.rotated_rect())
    order = (0, 1, 3) if not region.vertical else (2, 3, 1)
    source = np.float32([corners[index] for index in order])
    destination = np.float32([[0, 0], [target_width, 0], [0, TARGET_HEIGHT]])
    matrix = cv2.getAffineTransform(source, destination)
    return cv2.warpAffine(
        rgb,
        matrix,
        (target_width, TARGET_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def recognise_patch(
    runtime,
    net,
    patch: np.ndarray,
    characters,
    blobs: tuple[str, str] = ("in0", "out0"),
    offset: int = 1,
) -> tuple[str, float]:
    """Run the recognition net on one patch and greedy-decode the CTC output."""
    height, width = patch.shape[:2]
    mat = runtime.Mat.from_pixels(
        np.ascontiguousarray(patch), runtime.Mat.PixelType.PIXEL_RGB2BGR, width, height
    )
    mat.substract_mean_normalize(MEAN, NORM)
    extractor = net.create_extractor()
    try:
        extractor.input(blobs[0], mat)
        code, out = extractor.extract(blobs[1])
        if code != 0:
            raise RuntimeError("recognition extraction failed")
        logits = np.array(out)
    finally:
        del extractor
    if logits.ndim == 3:
        logits = logits[0]
    return decode_ctc(logits, characters, offset)


def decode_ctc(logits: np.ndarray, characters, offset: int = 1) -> tuple[str, float]:
    """Greedy CTC decode: argmax per step, collapse repeats, skip blanks.

    ``offset`` reconciles the two dictionary conventions in circulation. The
    nihui PP-OCRv5 dictionary lists only real characters, so class ``i`` is
    ``characters[i - 1]`` (offset 1). The Avafly PP-OCRv6 keys files carry the
    CTC blank as their own first line, so class ``i`` is ``characters[i]``
    (offset 0). Getting this wrong shifts every character by one and produces
    fluent-looking nonsense, so it is stated per model rather than guessed.
    """
    indices = logits.argmax(axis=1)
    scores = logits.max(axis=1)
    pieces: list[str] = []
    confidences: list[float] = []
    last = 0
    for index, score in zip(indices, scores, strict=True):
        index = int(index)
        if index == last:
            continue
        last = index
        if index <= 0:
            continue
        position = index - offset
        if position < len(characters):
            pieces.append(characters[position])
            confidences.append(float(score))
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return "".join(pieces), confidence
