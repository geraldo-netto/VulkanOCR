"""Versioned corpus manifest contract and generator integration."""

import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from corpus_schema import load_manifest, validate_manifest  # noqa: E402, I001
from make_corpus import main as make_corpus  # noqa: E402, I001
from make_corpus import make_script_samples  # noqa: E402, I001


def _case() -> dict:
    return {
        "id": "latin-clean",
        "script": "Latn",
        "language": "en",
        "direction": "ltr",
        "lines": ["Exact line one", "Exact line two"],
        "font": {
            "family": "DejaVu Sans",
            "file": "DejaVuSans.ttf",
            "license": "Bitstream Vera Fonts Copyright",
        },
        "palette": {"foreground": "#000000", "background": "#FFFFFF"},
        "size": {"font_px": 28, "width_px": 900, "height_px": 140},
        "background_objects": [],
        "variant": "clean-28px-sans",
        "image": "latin-clean.png",
    }


def test_complete_version_one_manifest_is_valid():
    validate_manifest({"schema_version": 1, "cases": [_case()]})


def test_case_metadata_is_required_by_the_schema():
    document = {"schema_version": 1, "cases": [_case()]}
    del document["cases"][0]["background_objects"]

    with pytest.raises(ValueError, match="background_objects"):
        validate_manifest(document)


def test_unknown_manifest_version_is_refused(tmp_path):
    path = tmp_path / "ground-truth.json"
    path.write_text('{"schema_version": 2, "cases": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="schema version 1"):
        load_manifest(path)


def test_duplicate_case_ids_are_refused():
    duplicate = deepcopy(_case())
    with pytest.raises(ValueError, match="case ids must be unique"):
        validate_manifest({"schema_version": 1, "cases": [_case(), duplicate]})


def test_generator_writes_a_schema_valid_manifest(tmp_path):
    assert make_corpus(tmp_path) == 0

    document = load_manifest(tmp_path / "ground-truth.json")
    validate_manifest(document)
    assert len(document["cases"]) == 55
    assert all((tmp_path / case["image"]).is_file() for case in document["cases"])


def test_script_sample_generator_writes_exact_russian_metadata(tmp_path):
    assert make_script_samples(tmp_path) == 0

    document = load_manifest(tmp_path / "ground-truth.json")
    validate_manifest(document)
    russian = next(case for case in document["cases"] if case["language"] == "ru")
    assert russian["script"] == "Cyrl"
    assert russian["font"]["license"] == "SIL Open Font License 1.1"
    assert "Ёж" in russian["lines"][1]
    assert (tmp_path / russian["image"]).is_file()
    greek = next(case for case in document["cases"] if case["language"] == "el")
    assert greek["script"] == "Grek"
    assert "ΐ, ΰ" in greek["lines"][1]
    assert (tmp_path / greek["image"]).is_file()
    japanese = next(case for case in document["cases"] if case["language"] == "ja")
    assert japanese["script"] == "Jpan"
    assert "ひらがな、カタカナ、漢字" in japanese["lines"][0]
    assert (tmp_path / japanese["image"]).is_file()
    mandarin = next(case for case in document["cases"] if case["language"] == "zh")
    assert mandarin["script"] == "Hani"
    assert "简体中文" in mandarin["lines"][0]
    assert "繁體中文" in mandarin["lines"][1]
    assert (tmp_path / mandarin["image"]).is_file()
    arabic = next(case for case in document["cases"] if case["language"] == "ar")
    assert arabic["script"] == "Arab"
    assert arabic["direction"] == "rtl"
    assert "سَأَقْرَأُ" in arabic["lines"][1]
    assert (tmp_path / arabic["image"]).is_file()
    hebrew = next(case for case in document["cases"] if case["language"] == "he")
    assert hebrew["script"] == "Hebr"
    assert hebrew["direction"] == "rtl"
    assert "שלום, עולם!" in hebrew["lines"][1]
    assert (tmp_path / hebrew["image"]).is_file()
