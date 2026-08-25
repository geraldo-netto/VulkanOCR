"""Text detection: DB probability map to oriented boxes.

Preprocessing starts from nihui/ncnn-android-ppocrv5 ``ppocrv5.cpp``
(BSD 3-Clause, Tencent). Geometry corrections are documented beside their
implementations, including restoration of DB's unclip rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .inference import extract_output

MEAN = (0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0)
NORM = (1 / 0.229 / 255.0, 1 / 0.224 / 255.0, 1 / 0.225 / 255.0)
STRIDE = 32
PAD_VALUE = 114.0
BINARY_THRESHOLD = 0.3
BOX_THRESHOLD = 0.6
# DB's own unclip rule: a box is pushed out by `area * ratio / perimeter`,
# which for a text line is about 0.75 of its height. The reference port used
# a flat 1.95 enlargement instead — about half as much margin — and that
# shaved ascenders, accents and last glyphs off small text (`noite` read as
# `noíte`); switching to the real rule closed the accuracy gap to upstream.
# `unclip_offset` computes it exactly for the rectangles `minAreaRect`
# produces, no polygon clipper needed.
UNCLIP_RATIO = 1.5
MIN_SIZE_FACTOR = 3.0


class DetectionOutputError(ValueError):
    """The detector returned a tensor that is not its declared probability map."""


@dataclass(frozen=True, slots=True)
class TextRegion:
    """One detected text line as an oriented rectangle in image coordinates."""

    center_x: float
    center_y: float
    width: float
    height: float
    angle: float
    vertical: bool
    score: float

    def rotated_rect(self):
        return ((self.center_x, self.center_y), (self.width, self.height), self.angle)


def detect_regions(
    runtime, net, rgb: np.ndarray, target_size: int, blobs: tuple[str, str] = ("in0", "out0")
) -> list[TextRegion]:
    """Run the detection net and return oriented text regions.

    ``blobs`` names the input and output tensors: the nihui PP-OCRv5 ports use
    ``in0``/``out0``, the Avafly PP-OCRv6 ports use ``input``/``output``.
    """
    image_height, image_width = rgb.shape[:2]
    width, height, _scale = _scaled(image_width, image_height, target_size)
    # Each axis maps back through the ratio it was actually resized by. The
    # minor axis is truncated to an integer above, so its true ratio differs
    # from `scale`; mapping y through the x ratio drifted boxes ~5 px on a
    # 4001x4000 image.
    scale_x = width / image_width
    scale_y = height / image_height

    mat = runtime.Mat.from_pixels_resize(
        np.ascontiguousarray(rgb),
        runtime.Mat.PixelType.PIXEL_RGB2BGR,
        image_width,
        image_height,
        width,
        height,
    )
    wpad = -width % STRIDE
    hpad = -height % STRIDE
    padded = runtime.Mat()
    runtime.copy_make_border(
        mat,
        padded,
        hpad // 2,
        hpad - hpad // 2,
        wpad // 2,
        wpad - wpad // 2,
        runtime.BorderType.BORDER_CONSTANT,
        PAD_VALUE,
    )
    padded.substract_mean_normalize(MEAN, NORM)

    output = extract_output(net, padded, blobs, stage="detection")
    probability = _probability_map(output, height + hpad, width + wpad)

    return _regions(probability, scale_x, scale_y, wpad, hpad)


def _probability_map(output, expected_height: int, expected_width: int) -> np.ndarray:
    """Validate and unwrap the detector's ``1 x height x width`` output."""
    tensor = np.asarray(output)
    expected = (1, expected_height, expected_width)
    if tensor.shape != expected:
        raise DetectionOutputError(
            f"detection output has shape {tensor.shape}; expected probability map {expected}"
        )
    return tensor[0]


def _scaled(width: int, height: int, target_size: int) -> tuple[int, int, float]:
    scale = 1.0
    if max(width, height) > target_size:
        if width > height:
            scale = target_size / width
            return target_size, max(int(height * scale), 1), scale
        scale = target_size / height
        return max(int(width * scale), 1), target_size, scale
    return width, height, scale


def _regions(
    probability: np.ndarray,
    scale_x: float,
    scale_y: float,
    wpad: int,
    hpad: int,
) -> list[TextRegion]:
    bitmap = (probability > BINARY_THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    # Every contour is considered. The reference bounded this at 1000, and
    # with OpenCV's bottom-to-top contour order the slice threw away the top
    # of exactly the noisy pages that exceed it; scoring a contour is a mask
    # fill and a mean, cheap enough that the thresholds can do the rejecting.
    for contour in contours:
        if len(contour) <= 2:
            continue
        score = _contour_score(probability, contour)
        if score < BOX_THRESHOLD:
            continue
        (cx, cy), (rw, rh), angle = _original_rect(contour, scale_x, scale_y, wpad, hpad)
        if max(rw, rh) < MIN_SIZE_FACTOR:
            continue
        rw, rh, angle, vertical = _oriented(rw, rh, angle)
        offset = unclip_offset(rw, rh)
        rw += 2.0 * offset
        rh += 2.0 * offset
        regions.append(
            TextRegion(
                center_x=cx,
                center_y=cy,
                width=rw,
                height=rh,
                angle=angle,
                vertical=vertical,
                score=score,
            )
        )
    return regions


def _original_rect(
    contour: np.ndarray,
    scale_x: float,
    scale_y: float,
    wpad: int,
    hpad: int,
):
    """Map a padded probability-map contour before fitting its rectangle.

    Integer resize dimensions make the x and y ratios differ, sometimes
    sharply for extreme aspect ratios. Mapping corners after fitting cannot
    preserve angle or side lengths under that anisotropic transform.
    """
    original = np.asarray(contour, dtype=np.float32).copy()
    original[..., 0] = (original[..., 0] - wpad // 2) / scale_x
    original[..., 1] = (original[..., 1] - hpad // 2) / scale_y
    return cv2.minAreaRect(original)


def unclip_offset(short: float, long: float, ratio: float = UNCLIP_RATIO) -> float:
    """How far out a detected box is pushed, by DB's own rule.

    ``area * ratio / perimeter`` is what PaddleOCR's unclip applies through a
    polygon offset; for a rectangle the offset is that distance on every side,
    so the whole clipper dependency reduces to this one expression.
    """
    perimeter = 2.0 * (short + long)
    if perimeter <= 0.0:
        return 0.0
    return short * long * ratio / perimeter


def _oriented(rw: float, rh: float, angle: float) -> tuple[float, float, float, bool]:
    """Normalise a ``minAreaRect`` so the first side is the text's thickness.

    The reference implementation normalised from the raw angle alone, and its
    thresholds leave a gap: with OpenCV's ``[-90, 0)`` angle convention a line
    skewed 30°–60° keeps the *long* side first, so ``crop_region`` computed a
    48-high strip a few pixels wide and the band read as confident nonsense —
    45° returned ``['1']``. The invariant callers rely on is enforced
    unconditionally at the end instead: whatever the branches decided, the
    short side comes first.
    """
    vertical = False
    if -30 <= angle <= 30 and rh > rw * 2.7:
        vertical = True
    if (angle <= -60 or angle >= 60) and rw > rh * 2.7:
        vertical = True
    if angle < -30:
        angle += 180
    if not vertical and angle < 30:
        angle += 90
        rw, rh = rh, rw
    if vertical and angle >= 60:
        angle -= 90
        rw, rh = rh, rw
    if not vertical and rw > rh:
        angle -= 90
        rw, rh = rh, rw
    return rw, rh, angle, vertical


def _contour_score(probability: np.ndarray, contour: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    x, y = max(x, 0), max(y, 0)
    w = min(w, probability.shape[1] - x)
    h = min(h, probability.shape[0] - y)
    if w <= 0 or h <= 0:
        return 0.0
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour - np.array([x, y])], 255)
    return float(cv2.mean(probability[y : y + h, x : x + w], mask=mask)[0])
