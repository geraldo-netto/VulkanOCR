"""Text detection: DB probability map to oriented boxes.

Constants and geometry follow the reference implementation in
nihui/ncnn-android-ppocrv5 ``ppocrv5.cpp`` (BSD 3-Clause, Tencent), which
replaces the original DB unclip with a fixed box enlargement.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MEAN = (0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0)
NORM = (1 / 0.229 / 255.0, 1 / 0.224 / 255.0, 1 / 0.225 / 255.0)
STRIDE = 32
PAD_VALUE = 114.0
BINARY_THRESHOLD = 0.3
BOX_THRESHOLD = 0.6
ENLARGE_RATIO = 1.95
MIN_SIZE_FACTOR = 3.0
MAX_CANDIDATES = 1000


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
    width, height, scale = _scaled(image_width, image_height, target_size)

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

    extractor = net.create_extractor()
    try:
        extractor.input(blobs[0], padded)
        code, out = extractor.extract(blobs[1])
        if code != 0:
            raise RuntimeError("detection extraction failed")
        probability = np.array(out)[0]
    finally:
        del extractor

    return _regions(probability, scale, wpad, hpad)


def _scaled(width: int, height: int, target_size: int) -> tuple[int, int, float]:
    scale = 1.0
    if max(width, height) > target_size:
        if width > height:
            scale = target_size / width
            return target_size, max(int(height * scale), 1), scale
        scale = target_size / height
        return max(int(width * scale), 1), target_size, scale
    return width, height, scale


def _regions(probability: np.ndarray, scale: float, wpad: int, hpad: int) -> list[TextRegion]:
    bitmap = (probability > BINARY_THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours[:MAX_CANDIDATES]:
        if len(contour) <= 2:
            continue
        score = _contour_score(probability, contour)
        if score < BOX_THRESHOLD:
            continue
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
        if max(rw, rh) < MIN_SIZE_FACTOR * scale:
            continue
        rw, rh, angle, vertical = _oriented(rw, rh, angle)
        rh += rw * (ENLARGE_RATIO - 1.0)
        rw *= ENLARGE_RATIO
        regions.append(
            TextRegion(
                center_x=(cx - wpad // 2) / scale,
                center_y=(cy - hpad // 2) / scale,
                width=rw / scale,
                height=rh / scale,
                angle=angle,
                vertical=vertical,
                score=score,
            )
        )
    return regions


def _oriented(rw: float, rh: float, angle: float) -> tuple[float, float, float, bool]:
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
