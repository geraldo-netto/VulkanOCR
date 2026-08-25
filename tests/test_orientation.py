"""PP-LCNet orientation preprocessing and correction without ncnn."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vulkanocr.detection import TextRegion, _oriented
from vulkanocr.orientation import (
    classify_patch_orientation,
    prepare_orientation_patch,
    rotate_patch,
)
from vulkanocr.recognition import crop_region


class FakeMat:
    PixelType = SimpleNamespace(PIXEL_RGB2BGR=1)
    received = None

    @classmethod
    def from_pixels(cls, pixels, _pixel_type, width, height):
        cls.received = (pixels.copy(), width, height)
        return cls()

    def substract_mean_normalize(self, mean, norm):
        self.normalization = (mean, norm)


class FakeNet:
    def __init__(self, scores=(0.1, 0.9), code=0):
        self.scores = scores
        self.code = code

    def create_extractor(self):
        net = self

        class Extractor:
            def input(self, name, mat):
                self.input_call = (name, mat)

            def extract(self, name):
                self.output_name = name
                return net.code, np.array(net.scores, dtype=np.float32)

        return Extractor()


def test_classifier_prepares_160_by_80_and_selects_180_degrees():
    runtime = SimpleNamespace(Mat=FakeMat)
    patch = np.zeros((48, 72, 3), dtype=np.uint8)

    degrees, confidence = classify_patch_orientation(runtime, FakeNet(), patch)

    assert FakeMat.received is not None
    prepared, width, height = FakeMat.received
    assert prepared.shape == (80, 160, 3)
    assert (width, height) == (160, 80)
    assert degrees == 180
    assert confidence == pytest.approx(0.9)


def test_180_degree_correction_reverses_both_patch_axes():
    patch = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert np.array_equal(rotate_patch(patch, 180), patch[::-1, ::-1])
    assert rotate_patch(patch, 0) is patch


@pytest.mark.parametrize("width", [24, 120, 400])
def test_smart_resize_always_returns_classifier_shape(width):
    prepared = prepare_orientation_patch(np.zeros((48, width, 3), dtype=np.uint8))
    assert prepared.shape == (80, 160, 3)


def _detected_text_region(rgb: np.ndarray) -> TextRegion:
    mask = np.any(rgb < 200, axis=2).astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = np.concatenate(contours)
    (center_x, center_y), (width, height), angle = cv2.minAreaRect(points)
    width, height, angle, vertical = _oriented(width, height, angle)
    return TextRegion(center_x, center_y, width + 15, height + 15, angle, vertical, 1.0)


@pytest.mark.parametrize(
    ("degrees", "quarter_turns", "correction"),
    [(0, 0, 0), (90, 3, 0), (180, 2, 180), (270, 1, 180)],
)
def test_cardinal_page_crops_become_upright_with_binary_correction(
    degrees, quarter_turns, correction
):
    page = np.full((180, 760, 3), 255, dtype=np.uint8)
    cv2.putText(
        page,
        "Vulkan rotate 1234",
        (35, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    turned = np.rot90(page, quarter_turns).copy()
    patch = crop_region(turned, _detected_text_region(turned))
    reference = crop_region(page, _detected_text_region(page))
    patch = cv2.resize(patch, (reference.shape[1], reference.shape[0]))

    corrected = rotate_patch(patch, correction)
    wrong = rotate_patch(patch, 180 - correction)
    corrected_error = np.mean(np.abs(corrected.astype(np.float32) - reference))
    wrong_error = np.mean(np.abs(wrong.astype(np.float32) - reference))

    assert corrected_error < wrong_error, degrees
