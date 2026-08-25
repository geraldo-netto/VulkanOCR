"""Geometry-only whitespace reconstruction, independent of CTC decoding."""

import pytest

from vulkanocr.detection import TextRegion
from vulkanocr.engine import assemble_result


def _region(
    center_x: float,
    center_y: float,
    *,
    length: float = 40.0,
    thickness: float = 20.0,
) -> TextRegion:
    return TextRegion(
        center_x=center_x,
        center_y=center_y,
        width=thickness,
        height=length,
        angle=90.0,
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
