"""The one assembler both engines share (VOCR-0041)."""

from vulkanocr.detection import TextRegion
from vulkanocr.engine import assemble_result


def _region(x: float, y: float) -> TextRegion:
    return TextRegion(
        center_x=x, center_y=y, width=10.0, height=100.0, angle=90.0, vertical=False, score=0.9
    )


def test_lines_sort_by_row_then_column_and_undecoded_regions_are_counted():
    result = assemble_result(
        "GPU A + GPU B",
        (
            (_region(300.0, 50.0), ("right", 0.8)),
            (_region(10.0, 200.0), ("below", 0.9)),
            (_region(20.0, 50.0), ("left", 0.7)),
            (_region(99.0, 99.0), ("", 0.0)),
        ),
    )
    assert result.device_name == "GPU A + GPU B"
    assert [line.text for line in result.lines] == ["left", "right", "below"]
    # The detector saw four regions; one defeated the recogniser, and the
    # result says so instead of printing identically to a clean page.
    assert result.undecoded_regions == 1
    line = result.lines[0]
    assert (line.center_x, line.center_y) == (20.0, 50.0)
    assert (line.thickness, line.length) == (10.0, 100.0)
    assert (line.box_score, line.confidence) == (0.9, 0.7)


def test_an_empty_page_assembles_to_an_empty_result():
    result = assemble_result("GPU", ())
    assert result.lines == ()
    assert result.undecoded_regions == 0
