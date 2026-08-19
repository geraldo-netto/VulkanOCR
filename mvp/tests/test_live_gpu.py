"""End-to-end on the real Vulkan device; skipped when hardware is absent."""

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
ncnn = pytest.importorskip("ncnn")

from ocr_engine import (
    CATALOG,
    DEFAULT_MODEL,
    HardwareVulkanUnavailable,
    OcrEngine,
    OcrEngineError,
    OcrModels,
    models_for,
)
from ocr_engine.device import select_hardware_device

try:
    select_hardware_device(ncnn)
    HARDWARE = True
except HardwareVulkanUnavailable:
    HARDWARE = False

needs_gpu = pytest.mark.skipif(not HARDWARE, reason="no hardware Vulkan device")


def models() -> OcrModels:
    return models_for()


def render(text: str) -> np.ndarray:
    image = np.full((80, 640, 3), 255, dtype=np.uint8)
    cv2.putText(image, text, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    return image


@pytest.fixture(scope="module")
def engine() -> OcrEngine:
    if not HARDWARE:
        pytest.skip("no hardware Vulkan device")
    return OcrEngine(models())


@needs_gpu
def test_reads_rendered_text_exactly(engine):
    result = engine.read(render("Vulkan 1234"))
    assert [line.text for line in result.lines] == ["Vulkan 1234"]
    assert result.lines[0].confidence > 0.8
    assert "llvmpipe" not in result.device_name


@needs_gpu
def test_blank_image_reads_zero_lines(engine):
    result = engine.read(np.full((120, 320, 3), 255, dtype=np.uint8))
    assert result.lines == ()


@needs_gpu
def test_line_coordinates_land_on_the_text(engine):
    image = np.full((200, 640, 3), 255, dtype=np.uint8)
    cv2.putText(image, "corner", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    result = engine.read(image)
    assert len(result.lines) == 1
    line = result.lines[0]
    assert line.text == "corner"
    assert line.center_y < 100  # top half of the image
    assert line.center_x < 320  # left half of the image


def test_missing_model_is_a_stable_refusal(tmp_path):
    broken = OcrModels(
        det_param=tmp_path / "absent.param",
        rec_param=tmp_path / "absent.param",
        dictionary=tmp_path / "absent.txt",
    )
    with pytest.raises(OcrEngineError) as caught:
        broken.validated()
    assert caught.value.code == "model-missing"


@needs_gpu
def test_invalid_image_is_a_stable_refusal(engine):
    with pytest.raises(OcrEngineError) as caught:
        engine.read(np.zeros((4, 4), dtype=np.uint8))
    assert caught.value.code == "image-invalid"
