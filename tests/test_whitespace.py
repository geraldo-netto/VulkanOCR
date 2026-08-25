"""Geometry-only whitespace reconstruction, independent of CTC decoding."""

import math

import pytest

from vulkanocr.detection import TextRegion
from vulkanocr.engine import assemble_result


def _region(
    center_x: float,
    center_y: float,
    *,
    length: float = 40.0,
    thickness: float = 20.0,
    angle: float = 90.0,
) -> TextRegion:
    return TextRegion(
        center_x=center_x,
        center_y=center_y,
        width=thickness,
        height=length,
        angle=angle,
        vertical=False,
        score=0.9,
    )


def test_assembly_joins_collinear_regions_with_a_geometry_inferred_space():
    result = assemble_result(
        "GPU",
        [
            (_region(20.0, 50.0), ("hello", 0.8)),
            (_region(70.0, 50.0), ("world", 1.0)),
        ],
    )

    assert len(result.lines) == 1
    line = result.lines[0]
    assert line.text == "hello world"
    assert (line.center_x, line.center_y) == pytest.approx((45.0, 50.0))
    assert (line.length, line.thickness) == pytest.approx((90.0, 20.0))
    assert line.confidence == pytest.approx(0.9)


def test_regions_on_separate_rows_remain_separate_lines():
    result = assemble_result(
        "GPU",
        [
            (_region(20.0, 30.0), ("upper", 0.9)),
            (_region(20.0, 70.0), ("lower", 0.9)),
        ],
    )

    assert [line.text for line in result.lines] == ["upper", "lower"]


@pytest.mark.parametrize(
    ("left_text", "left_length", "right_text", "right_length"),
    [
        pytest.param("WWW", 54.0, "iii", 18.0, id="proportional"),
        pytest.param("code", 40.0, "block", 50.0, id="monospace"),
    ],
)
def test_font_metrics_do_not_change_the_thickness_scaled_gap_rule(
    left_text, left_length, right_text, right_length
):
    gap = 10.0
    left_center = left_length / 2.0
    right_center = left_length + gap + right_length / 2.0

    result = assemble_result(
        "GPU",
        [
            (_region(left_center, 50.0, length=left_length), (left_text, 0.9)),
            (_region(right_center, 50.0, length=right_length), (right_text, 0.9)),
        ],
    )

    assert [line.text for line in result.lines] == [f"{left_text} {right_text}"]


def test_opening_closing_and_terminal_punctuation_do_not_gain_spaces():
    fragments = [
        ("(", 5.0, 10.0),
        ("value", 40.0, 40.0),
        (",", 74.0, 8.0),
        ("next", 104.0, 32.0),
        (")", 134.0, 8.0),
    ]

    result = assemble_result(
        "GPU",
        [(_region(x, 50.0, length=length), (text, 0.9)) for text, x, length in fragments],
    )

    assert [line.text for line in result.lines] == ["(value, next)"]


def test_a_wide_inter_region_gap_can_infer_multiple_spaces():
    result = assemble_result(
        "GPU",
        [
            (_region(20.0, 50.0), ("left", 0.9)),
            (_region(90.0, 50.0), ("right", 0.9)),
        ],
    )

    assert [line.text for line in result.lines] == ["left   right"]


def test_existing_multiple_spaces_are_preserved_without_normalisation():
    result = assemble_result(
        "GPU",
        [
            (_region(20.0, 50.0), ("left  ", 0.9)),
            (_region(70.0, 50.0), ("right", 0.9)),
        ],
    )

    assert [line.text for line in result.lines] == ["left  right"]


def test_rotated_fragments_use_their_local_reading_axis():
    angle = 45.0
    axis = (math.sin(math.radians(angle)), -math.cos(math.radians(angle)))
    left_center = (100.0, 100.0)
    right_center = (left_center[0] + 50.0 * axis[0], left_center[1] + 50.0 * axis[1])

    result = assemble_result(
        "GPU",
        [
            (_region(*left_center, angle=angle), ("skew", 0.9)),
            (_region(*right_center, angle=angle), ("line", 0.9)),
        ],
    )

    assert len(result.lines) == 1
    line = result.lines[0]
    assert line.text == "skew line"
    assert (line.center_x, line.center_y) == pytest.approx(
        (left_center[0] + 25.0 * axis[0], left_center[1] + 25.0 * axis[1])
    )
    assert (line.length, line.angle) == pytest.approx((90.0, angle))


def test_rtl_regions_are_joined_right_to_left_without_reversing_characters():
    result = assemble_result(
        "GPU",
        [
            (_region(80.0, 50.0), ("שלום", 0.9)),
            (_region(30.0, 50.0), ("עולם", 0.9)),
            (_region(1.0, 50.0, length=8.0), ("!", 0.9)),
        ],
    )

    assert [line.text for line in result.lines] == ["שלום עולם!"]


def test_neutral_punctuation_cannot_bridge_mixed_strong_directions():
    result = assemble_result(
        "GPU",
        [
            (_region(20.0, 50.0), ("left", 0.9)),
            (_region(70.0, 50.0, length=8.0), (".", 0.9)),
            (_region(120.0, 50.0), ("ימין", 0.9)),
        ],
    )

    assert [line.text for line in result.lines] == ["left", ".", "ימין"]


def test_duplicate_or_strongly_overlapping_regions_are_not_concatenated():
    region = _region(20.0, 50.0, length=90.0, thickness=10.0)

    result = assemble_result("GPU", [(region, ("same", 0.9)), (region, ("same", 0.9))])

    assert [line.text for line in result.lines] == ["same", "same"]
