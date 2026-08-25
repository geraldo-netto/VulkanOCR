"""The one assembler both engines share (VOCR-0041)."""

import json
from pathlib import Path

from vulkanocr.detection import TextRegion
from vulkanocr.engine import assemble_result
from vulkanocr.policy import FalsePositivePolicy, RecognitionContext


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


def test_labeled_icon_readings_are_filtered_but_labeled_cjk_text_is_retained():
    manifest = Path(__file__).resolve().parents[1] / "samples/corpus/ground-truth.json"
    cases = json.loads(manifest.read_text(encoding="utf-8"))["cases"]
    negatives = [case for case in cases if case["content_label"].startswith("negative-")]
    positive = next(case for case in cases if case["content_label"] == "positive-cjk")
    readings = frozenset(reading for case in negatives for reading in case["known_false_readings"])
    policy = FalsePositivePolicy(readings, below_confidence=0.6)
    recognised = [
        (_region(float(index), 20.0), (reading, 0.4))
        for index, reading in enumerate(sorted(readings))
    ]

    icons = assemble_result(
        "GPU",
        recognised,
        false_positive_policy=policy,
        recognition_context=RecognitionContext(page_languages=frozenset({"en"})),
    )
    cjk = assemble_result(
        "GPU",
        [
            (_region(float(index * 200), 20.0), (line, 0.4))
            for index, line in enumerate(positive["lines"])
        ],
        false_positive_policy=policy,
        recognition_context=RecognitionContext(page_languages=frozenset({"zh"})),
    )

    assert icons.lines == ()
    assert icons.filtered_regions == len(readings)
    assert [line.text for line in cjk.lines] == positive["lines"]
    assert cjk.filtered_regions == 0
