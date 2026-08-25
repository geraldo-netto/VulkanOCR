"""Versioned corpus manifest contract and generator integration."""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageColor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from corpus_schema import (  # noqa: E402, I001
    load_manifest,
    require_engine_selections,
    validate_manifest,
)
import make_corpus as corpus_generator  # noqa: E402, I001
from make_corpus import CorpusWriter  # noqa: E402, I001
from make_corpus import main as make_corpus  # noqa: E402, I001
from make_corpus import make_orientation_samples  # noqa: E402, I001
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
        "recognition": {
            "vulkanocr": {"model": "v6-medium"},
            "paddleocr": {"language": "en", "model": "PP-OCRv6"},
            "tesseract": {"language": "eng"},
        },
        "variant": "clean-28px-sans",
        "image": "latin-clean.png",
    }


def test_complete_version_two_manifest_is_valid():
    validate_manifest({"schema_version": 2, "cases": [_case()]})


def test_case_metadata_is_required_by_the_schema():
    document = {"schema_version": 2, "cases": [_case()]}
    del document["cases"][0]["background_objects"]

    with pytest.raises(ValueError, match="background_objects"):
        validate_manifest(document)


def test_unknown_manifest_version_is_refused(tmp_path):
    path = tmp_path / "ground-truth.json"
    path.write_text('{"schema_version": 3, "cases": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="schema version 2"):
        load_manifest(path)


def test_duplicate_case_ids_are_refused():
    duplicate = deepcopy(_case())
    with pytest.raises(ValueError, match="case ids must be unique"):
        validate_manifest({"schema_version": 2, "cases": [_case(), duplicate]})


def test_runner_refuses_cases_without_its_declared_model():
    unsupported = _case()
    unsupported["recognition"]["vulkanocr"] = None

    with pytest.raises(ValueError, match="no vulkanocr model.*latin-clean"):
        require_engine_selections([unsupported], "vulkanocr")


def test_generator_writes_a_schema_valid_manifest(tmp_path):
    assert make_corpus([str(tmp_path)]) == 0

    document = load_manifest(tmp_path / "ground-truth.json")
    validate_manifest(document)
    assert len(document["cases"]) == 55
    assert all((tmp_path / case["image"]).is_file() for case in document["cases"])


def test_main_requires_an_output_path():
    with pytest.raises(SystemExit):
        make_corpus([])


def test_writer_refuses_a_failed_image_write(tmp_path, monkeypatch):
    writer = CorpusWriter(tmp_path)
    monkeypatch.setattr(corpus_generator.cv2, "imwrite", lambda *_args: False)

    with pytest.raises(OSError, match="cannot write corpus image"):
        writer.add(_case(), np.zeros((2, 2, 3), dtype=np.uint8))

    assert writer.cases == []
    assert not (tmp_path / "ground-truth.json").exists()


def test_generator_validates_fonts_before_writing(tmp_path, monkeypatch):
    monkeypatch.setitem(corpus_generator.FONTS, "sans", str(tmp_path / "missing.ttf"))

    with pytest.raises(FileNotFoundError, match="missing.ttf"):
        make_corpus([str(tmp_path / "corpus")])

    assert not (tmp_path / "corpus").exists()


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
    georgian = next(case for case in document["cases"] if case["language"] == "ka")
    assert georgian["script"] == "Geor"
    assert "ქართული ტექსტი" in georgian["lines"][1]
    assert (tmp_path / georgian["image"]).is_file()
    azerbaijani = next(case for case in document["cases"] if case["language"] == "az")
    assert azerbaijani["script"] == "Latn"
    for letter in "ƏĞİÖŞÜÇ":
        assert letter in azerbaijani["lines"][0]
    assert (tmp_path / azerbaijani["image"]).is_file()
    assert len(document["cases"]) == 24
    assert all("recognition" in case for case in document["cases"])
    assert {case["size"]["font_px"] for case in document["cases"]} == {28, 36, 44}
    assert len({tuple(case["palette"].values()) for case in document["cases"]}) == 3
    assert len({case["font"]["family"] for case in document["cases"]}) >= 8
    assert {item["kind"] for case in document["cases"] for item in case["background_objects"]} == {
        "circle",
        "rectangle",
        "polygon",
    }
    for case in document["cases"]:
        image = Image.open(tmp_path / case["image"]).convert("RGB")
        background = ImageColor.getrgb(case["palette"]["background"])
        width, height = image.size
        assert all(image.getpixel((x, 0)) == background for x in range(width))
        assert all(image.getpixel((x, height - 1)) == background for x in range(width))
        assert all(image.getpixel((0, y)) == background for y in range(height))
        assert all(image.getpixel((width - 1, y)) == background for y in range(height))


def test_orientation_generator_writes_all_cardinal_cases(tmp_path):
    assert make_orientation_samples(tmp_path) == 0

    document = load_manifest(tmp_path / "ground-truth.json")
    validate_manifest(document)
    assert {case["variant"] for case in document["cases"]} == {
        "orientation-0deg",
        "orientation-90deg",
        "orientation-180deg",
        "orientation-270deg",
    }
    assert all(
        case["recognition"]["vulkanocr"] == {"model": "v6-medium"} for case in document["cases"]
    )
    assert all((tmp_path / case["image"]).is_file() for case in document["cases"])
