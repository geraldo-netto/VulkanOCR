"""The geometry invariant recognition depends on, checked without a GPU.

`crop_region` computes its output width as `long / short * 48`, so if
`_oriented` ever returns the long side first the crop collapses to a sliver
and the band reads as confident nonsense — which is exactly what happened for
30°–60° skew before the invariant below was enforced (VOCR-0001).
"""

import numpy as np
import pytest

from vulkanocr.detection import TextRegion, _oriented, unclip_offset
from vulkanocr.recognition import crop_region


def test_the_short_side_always_comes_first():
    """Every angle OpenCV can produce, every aspect a text line can have.

    OpenCV 5's `minAreaRect` reports angles in [-90, 0); older releases used
    other conventions, so the grid runs the full circle to cover them all.
    """
    for angle_int in range(-180, 181, 1):
        for short, long in ((12.0, 40.0), (20.0, 300.0), (54.0, 129.0), (47.0, 1088.0)):
            for sides in ((short, long), (long, short)):
                out_w, out_h, _angle, vertical = _oriented(sides[0], sides[1], float(angle_int))
                if not vertical:
                    assert out_w <= out_h, (sides, angle_int, out_w, out_h)


def test_the_45_degree_band_yields_a_real_crop_not_a_sliver():
    """The exact triple measured from a 45° line before the fix: (129.4, 53.7, -45)."""
    out_w, out_h, _angle, _vertical = _oriented(129.4, 53.7, -45.0)
    region = TextRegion(
        center_x=450.0,
        center_y=150.0,
        width=out_w,
        height=out_h,
        angle=_angle,
        vertical=False,
        score=0.9,
    )
    patch = crop_region(np.full((300, 900, 3), 255, np.uint8), region)

    assert patch.shape[0] == 48
    # 48 * long/short ≈ 115 columns; the bug produced 19.
    assert patch.shape[1] > 80, patch.shape


def test_unclip_grows_a_line_by_roughly_three_quarters_of_its_height():
    """DB's rule, area * ratio / perimeter, for a long thin line."""
    offset = unclip_offset(16.0, 400.0)
    assert 11.0 < offset < 12.0
    assert unclip_offset(0.0, 0.0) == 0.0


def test_a_horizontal_lines_bounding_box_is_wider_than_it_is_tall():
    """`width`/`height` were the rect's short/long sides: a horizontal line
    reported width 54, height 356, and an axis-aligned box drawn from them
    came out rotated a quarter turn (VOCR-0003)."""
    from vulkanocr.engine import OcrLine

    line = OcrLine(
        text="Vulkan 1234",
        confidence=0.9,
        box_score=0.9,
        center_x=240.0,
        center_y=52.0,
        thickness=54.0,
        length=356.0,
        angle=90.0,
        vertical=False,
    )
    left, top, right, bottom = line.bounding_box()

    assert (right - left) > (bottom - top)
    assert abs((right + left) / 2 - 240.0) < 1e-3
    assert abs((bottom + top) / 2 - 52.0) < 1e-3


def test_each_axis_maps_back_through_its_own_ratio():
    """A 4001x4000 image resizes to 640x639: the y axis's real ratio is
    639/4000, and mapping through the x ratio drifted the bottom ~5 px."""
    from vulkanocr.detection import _scaled

    width, height, _scale = _scaled(4001, 4000, 640)

    assert (width, height) == (640, 639)
    bottom_via_own = (height - 1) / (height / 4000)
    bottom_via_x = (height - 1) / (width / 4001)
    assert abs(bottom_via_x - bottom_via_own) > 4.0  # the bug's size
    assert abs(bottom_via_own - 3993.7) < 0.1  # its own ratio lands right


def test_rectangle_geometry_is_fitted_after_anisotropic_mapping():
    """A 10000x2 image resizes to 640x1, making x/y ratios 0.064/0.5."""
    from vulkanocr.detection import _original_rect

    contour = np.array([[[64, 15]], [[576, 15]], [[576, 16]], [[64, 16]]], np.float32)

    (cx, cy), (side_a, side_b), _angle = _original_rect(
        contour,
        scale_x=640 / 10_000,
        scale_y=1 / 2,
        wpad=0,
        hpad=31,
    )

    assert cx == pytest.approx(5000.0)
    assert cy == pytest.approx(1.0)
    assert sorted((side_a, side_b)) == pytest.approx([2.0, 8000.0])


def test_rotated_anisotropic_box_round_trips_into_original_coordinates():
    import cv2

    from vulkanocr.detection import TextRegion, _oriented, _original_rect

    original_rect = ((520.0, 170.0), (32.0, 420.0), 23.0)
    expected_corners = cv2.boxPoints(original_rect)
    scale_x, scale_y = 640 / 1001, 212 / 333
    wpad, hpad = 0, 12
    probability_corners = expected_corners.copy()
    probability_corners[:, 0] = probability_corners[:, 0] * scale_x + wpad // 2
    probability_corners[:, 1] = probability_corners[:, 1] * scale_y + hpad // 2
    contour = probability_corners.reshape((-1, 1, 2)).astype(np.float32)

    (cx, cy), (side_a, side_b), angle = _original_rect(
        contour,
        scale_x=scale_x,
        scale_y=scale_y,
        wpad=wpad,
        hpad=hpad,
    )

    assert (cx, cy) == pytest.approx(original_rect[0], abs=1e-3)
    assert sorted((side_a, side_b)) == pytest.approx([32.0, 420.0], abs=1e-3)
    restored_corners = cv2.boxPoints(((cx, cy), (side_a, side_b), angle))
    for corner in expected_corners:
        assert min(np.linalg.norm(corner - restored) for restored in restored_corners) < 1e-3

    width, height, angle, vertical = _oriented(side_a, side_b, angle)
    region = TextRegion(cx, cy, width, height, angle, vertical, score=0.9)
    image = np.full((333, 1001, 3), 255, np.uint8)
    cv2.fillConvexPoly(image, np.rint(expected_corners).astype(np.int32), (0, 0, 0))

    patch = crop_region(image, region)

    assert patch.shape[0] == 48
    assert patch.shape[1] == pytest.approx(630, abs=2)
    assert patch[12:-12, patch.shape[1] // 4 : patch.shape[1] * 3 // 4].mean() < 5


def test_a_trailing_newline_is_a_file_convention_not_a_character_class(tmp_path):
    """The Avafly clone's ppocr_keys_v5 ends with a newline; the nihui copy
    does not. Both must yield the same classes, and the final literal-space
    class must survive (VOCR-0017)."""
    from vulkanocr.engine import OcrEngine

    bare = tmp_path / "bare.txt"
    bare.write_bytes(b"a\nb\n ")
    newline = tmp_path / "newline.txt"
    newline.write_bytes(b"a\nb\n \n")

    read_bare = OcrEngine._load_dictionary(bare)
    read_newline = OcrEngine._load_dictionary(newline)

    assert read_bare == read_newline == ("a", "b", " ")


class TestScaledAndMapping:
    def test_small_images_are_not_upscaled(self):
        from vulkanocr.detection import _scaled

        assert _scaled(300, 200, 640) == (300, 200, 1.0)

    def test_the_long_side_lands_exactly_on_the_target(self):
        from vulkanocr.detection import _scaled

        for width, height in ((1280, 720), (720, 1280), (5000, 100)):
            out_w, out_h, scale = _scaled(width, height, 640)
            assert max(out_w, out_h) == 640
            assert min(out_w, out_h) >= 1
            assert scale == pytest.approx(640 / max(width, height))


class TestContourScore:
    def test_the_score_is_the_mean_probability_inside_the_contour(self):
        import cv2

        from vulkanocr.detection import _contour_score

        probability = np.zeros((40, 40), np.float32)
        probability[10:20, 10:30] = 0.8
        contours, _ = cv2.findContours(
            (probability > 0.3).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        assert _contour_score(probability, contours[0]) == pytest.approx(0.8, abs=0.05)


class TestCropRegion:
    def test_a_horizontal_region_rectifies_to_its_length_by_48(self):
        from vulkanocr.detection import TextRegion

        image = np.full((100, 400, 3), 255, np.uint8)
        image[40:60, 50:350] = 0  # a 300x20 bar
        region = TextRegion(
            center_x=200.0,
            center_y=50.0,
            width=20.0,
            height=300.0,
            angle=90.0,
            vertical=False,
            score=0.9,
        )

        patch = crop_region(image, region)

        assert patch.shape[0] == 48
        assert 600 < patch.shape[1] < 800  # 48 * 300/20 = 720
        assert patch[24, patch.shape[1] // 2].tolist() == [0, 0, 0]  # the bar is inside
