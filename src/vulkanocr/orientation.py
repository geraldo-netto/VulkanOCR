"""PP-LCNet text-line orientation preprocessing and binary classification."""

from __future__ import annotations

import cv2
import numpy as np

TARGET_WIDTH = 160
TARGET_HEIGHT = 80
MAX_DOWNSCALE = 3.0
PAD_VALUE = 114
MEAN = (127.5, 127.5, 127.5)
NORM = (1 / 127.5, 1 / 127.5, 1 / 127.5)


def prepare_orientation_patch(patch: np.ndarray) -> np.ndarray:
    """Match the Avafly PP-LCNet smart-resize policy exactly."""
    ratio = TARGET_HEIGHT / patch.shape[0]
    resized_width = int(patch.shape[1] * ratio)
    if resized_width < TARGET_WIDTH:
        resized = cv2.resize(patch, (resized_width, TARGET_HEIGHT))
        return cv2.copyMakeBorder(
            resized,
            0,
            0,
            0,
            TARGET_WIDTH - resized_width,
            cv2.BORDER_CONSTANT,
            value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
        )
    if resized_width < TARGET_WIDTH * MAX_DOWNSCALE:
        return cv2.resize(patch, (TARGET_WIDTH, TARGET_HEIGHT))
    crop_width = int(MAX_DOWNSCALE * TARGET_WIDTH / ratio)
    return cv2.resize(patch[:, :crop_width], (TARGET_WIDTH, TARGET_HEIGHT))


def classify_patch_orientation(
    runtime,
    net,
    patch: np.ndarray,
    blobs: tuple[str, str] = ("input", "output"),
    labels: tuple[int, int] = (0, 180),
) -> tuple[int, float]:
    """Return the selected clockwise correction and its softmax confidence."""
    prepared = prepare_orientation_patch(patch)
    mat = runtime.Mat.from_pixels(
        np.ascontiguousarray(prepared),
        runtime.Mat.PixelType.PIXEL_RGB2BGR,
        TARGET_WIDTH,
        TARGET_HEIGHT,
    )
    mat.substract_mean_normalize(MEAN, NORM)
    extractor = net.create_extractor()
    try:
        extractor.input(blobs[0], mat)
        code, out = extractor.extract(blobs[1])
        if code != 0:
            raise RuntimeError("orientation extraction failed")
        scores = np.asarray(out, dtype=np.float32).reshape(-1)
    finally:
        del extractor
    if scores.size != len(labels):
        raise RuntimeError(f"orientation output has {scores.size} classes; expected {len(labels)}")
    index = int(scores.argmax())
    return labels[index], float(scores[index])


def rotate_patch(patch: np.ndarray, degrees: int) -> np.ndarray:
    """Apply the binary classifier's correction without changing dimensions."""
    if degrees == 0:
        return patch
    if degrees == 180:
        return cv2.rotate(patch, cv2.ROTATE_180)
    raise ValueError(f"unsupported text-line orientation correction: {degrees}")


__all__ = ["classify_patch_orientation", "prepare_orientation_patch", "rotate_patch"]
