"""The catalogue: default model set and per-port facts."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from vulkanocr import CATALOG, DEFAULT_MODEL, OcrEngineError, OcrModels, models_for


def test_detector_and_recognizer_specs_own_independent_runtime_facts():
    from vulkanocr.catalog import DetectorSpec, RecognizerSpec

    detector = DetectorSpec("det.param", ("det-in", "det-out"))
    recognizer = RecognizerSpec(
        "rec.param",
        "keys.txt",
        ("rec-in", "rec-out"),
        dictionary_includes_blank=False,
        required_precision="int8",
    )

    assert detector.param == "det.param"
    assert detector.blobs == ("det-in", "det-out")
    assert detector.required_precision is None
    assert recognizer.param == "rec.param"
    assert recognizer.dictionary == "keys.txt"
    assert recognizer.blobs == ("rec-in", "rec-out")
    assert recognizer.ctc_offset == 1
    assert recognizer.required_precision == "int8"
    with pytest.raises(FrozenInstanceError):
        detector.param = "changed.param"


def test_named_components_preserve_each_upstream_port_convention():
    from vulkanocr.catalog import DETECTORS, RECOGNIZERS

    assert DETECTORS["v6-medium-det"].blobs == ("input", "output")
    assert RECOGNIZERS["v6-medium-rec"].dictionary_includes_blank is True
    assert DETECTORS["v5-mobile-det"].blobs == ("in0", "out0")
    assert RECOGNIZERS["v5-mobile-rec"].ctc_offset == 1


def test_the_default_is_the_current_generation():
    assert DEFAULT_MODEL == "v6-medium"
    assert models_for().blobs == ("input", "output")
    assert models_for().dictionary_includes_blank is True
    assert models_for().orientation_param == (
        models_for().det_param.parent / "PP_LCNet_x0_25_textline_ori.param"
    )
    assert models_for().orientation_blobs == ("input", "output")
    assert models_for().orientation_labels == (0, 180)


def test_the_previous_generation_keeps_its_own_conventions():
    v5 = models_for("v5-mobile")
    assert v5.blobs == ("in0", "out0")
    assert v5.dictionary_includes_blank is False
    assert v5.orientation_param is None


def test_every_catalogued_model_set_is_installed():
    """An environment check, honest about being one: a clean clone has no
    models (they are third-party ports, gitignored on purpose), and a red
    test on every fresh checkout teaches people to ignore red tests."""
    try:
        for name in CATALOG:
            models_for(name).validated()
    except OcrEngineError:
        pytest.skip("model ports are not fetched here; see THIRD-PARTY.md")


def test_every_orientation_catalog_entry_is_installed_when_models_are_fetched():
    try:
        for name in CATALOG:
            models = models_for(name)
            if models.orientation_param is not None:
                models.validated(("ori",))
    except OcrEngineError:
        pytest.skip("model ports are not fetched here; see THIRD-PARTY.md")


def test_orientation_validation_requires_both_graph_files(tmp_path):
    dictionary = tmp_path / "keys.txt"
    dictionary.write_text("a\n", encoding="utf-8")
    orientation = tmp_path / "orientation.param"
    orientation.write_text("param", encoding="utf-8")
    models = OcrModels(
        tmp_path / "det.param",
        tmp_path / "rec.param",
        dictionary,
        orientation_param=orientation,
    )

    with pytest.raises(OcrEngineError, match="orientation.bin"):
        models.validated(("ori",))

    orientation.with_suffix(".bin").write_bytes(b"weights")
    assert models.validated(("ori",)) is models


def test_an_unknown_model_set_names_the_known_ones():
    try:
        models_for("v9-imaginary")
    except ValueError as error:
        assert "v6-medium" in str(error)
    else:
        raise AssertionError("expected a refusal")


def test_the_ctc_offset_is_a_fact_of_the_port_not_of_the_caller():
    """Eight call sites used to derive it independently; now they read it."""
    from vulkanocr.engine import OcrModels

    with_blank = OcrModels(
        Path("d.param"), Path("r.param"), Path("k.txt"), dictionary_includes_blank=True
    )
    without = OcrModels(
        Path("d.param"), Path("r.param"), Path("k.txt"), dictionary_includes_blank=False
    )

    assert with_blank.ctc_offset == 0
    assert without.ctc_offset == 1


def test_the_engine_offers_its_halves_to_consumers_that_time_them_apart():
    """The benchmarks reached into `engine._det` and friends 65 times before
    these existed; a private rename silently broke four scripts."""
    from vulkanocr.engine import OcrEngine

    for name in ("detect", "crops", "recognise", "logits", "decode"):
        assert callable(getattr(OcrEngine, name)), name


def test_a_named_port_supplies_its_own_facts(tmp_path):
    """External consumers name the port instead of restating it (VOCR-0061)."""
    from vulkanocr import models_for_port

    models = models_for_port(
        "avafly-v6", tmp_path / "det.param", tmp_path / "rec.param", tmp_path / "keys.txt"
    )
    assert models.blobs == ("input", "output")
    assert models.dictionary_includes_blank is True
    assert models.ctc_offset == 0
    nihui = models_for_port("nihui-v5", tmp_path / "d", tmp_path / "r", tmp_path / "k")
    assert nihui.blobs == ("in0", "out0")
    assert nihui.ctc_offset == 1
    with pytest.raises(ValueError, match="unknown model port"):
        models_for_port("mystery-port", tmp_path / "d", tmp_path / "r", tmp_path / "k")


def test_the_catalog_and_the_port_facts_agree():
    """The clone-layout catalog and the port registry state the same facts."""
    from vulkanocr.catalog import PORT_FACTS

    for name, spec in CATALOG.items():
        port = "nihui-v5" if name.startswith("v5") else "avafly-v6"
        blobs, includes_blank = PORT_FACTS[port]
        assert spec.blobs == blobs, name
        assert spec.dictionary_includes_blank is includes_blank, name
