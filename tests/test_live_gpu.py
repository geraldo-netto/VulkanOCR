"""End-to-end on the real Vulkan device; skipped when hardware is absent."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
ncnn = pytest.importorskip("ncnn")

from vulkanocr import (
    HardwareVulkanUnavailableError,
    OcrEngine,
    OcrEngineError,
    OcrModels,
    models_for,
)
from vulkanocr.device import select_hardware_device

try:
    select_hardware_device(ncnn)
    HARDWARE = True
except HardwareVulkanUnavailableError:
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


def test_the_30_to_60_degree_band_reads_text_instead_of_noise(engine):
    """45° used to return `['1']`: the sides stayed swapped and the crop was
    a 19-px sliver (VOCR-0001). The model still loses edge glyphs at strong
    skew — upstream does too — so the bar is the digits, not perfection."""
    for degrees in (35, 45):
        page = np.full((300, 900, 3), 255, np.uint8)
        cv2.putText(page, "Vulkan 1234", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
        matrix = cv2.getRotationMatrix2D((450, 150), degrees, 1.0)
        turned = cv2.warpAffine(page, matrix, (900, 300), borderValue=(255, 255, 255))
        rgb = np.ascontiguousarray(turned[:, :, ::-1])

        text = " ".join(line.text for line in engine.read(rgb).lines)

        assert "1234" in text, (degrees, text)


def test_the_gpu_pool_reads_exactly_what_one_gpu_reads(engine):
    """Device count must never change the answer (VOCR-0034). On a machine
    with one hardware device the pool falls back to the single engine, so
    this holds everywhere it runs."""
    from vulkanocr.parallel import ParallelOcr

    rgb = np.ascontiguousarray(render("Vulkan pool 1234")[:, :, ::-1])
    single = [line.text for line in engine.read(rgb).lines]

    with ParallelOcr(models()) as pool:
        pooled = [line.text for line in pool.read(rgb).lines]

    assert pooled == single
