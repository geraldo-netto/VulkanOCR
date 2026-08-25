"""PP-LCNet orientation preprocessing and correction without ncnn."""

from types import SimpleNamespace

import numpy as np
import pytest

from vulkanocr.orientation import (
    classify_patch_orientation,
    prepare_orientation_patch,
    rotate_patch,
)


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
