"""Comparison refuses incomplete or mislabeled engine runs."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from compare_engines import validate_result_rows  # noqa: E402, I001


EXPECTED = {"clean-page": "clean", "blur-page": "blur"}


def test_exact_case_ids_and_variants_are_accepted():
    validate_result_rows(
        "engine",
        [
            {"id": "clean-page", "variant": "clean"},
            {"id": "blur-page", "variant": "blur"},
        ],
        EXPECTED,
    )


@pytest.mark.parametrize(
    ("rows", "problem"),
    [
        ([{"id": "clean-page", "variant": "clean"}], "missing ids"),
        (
            [
                {"id": "clean-page", "variant": "clean"},
                {"id": "blur-page", "variant": "blur"},
                {"id": "ghost", "variant": "clean"},
            ],
            "extra ids",
        ),
        (
            [
                {"id": "clean-page", "variant": "blur"},
                {"id": "blur-page", "variant": "blur"},
            ],
            "wrong variants",
        ),
        (
            [
                {"id": "clean-page", "variant": "clean"},
                {"id": "clean-page", "variant": "clean"},
                {"id": "blur-page", "variant": "blur"},
            ],
            "duplicate ids",
        ),
    ],
    ids=["missing", "extra", "variant", "duplicate"],
)
def test_case_set_mismatch_is_refused(rows, problem):
    with pytest.raises(ValueError, match=problem):
        validate_result_rows("engine", rows, EXPECTED)
