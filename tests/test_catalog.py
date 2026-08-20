"""The catalogue: default model set and per-port facts."""

from pathlib import Path

from vulkanocr import CATALOG, DEFAULT_MODEL, models_for


def test_the_default_is_the_current_generation():
    assert DEFAULT_MODEL == "v6-medium"
    assert models_for().blobs == ("input", "output")
    assert models_for().dictionary_includes_blank is True


def test_the_previous_generation_keeps_its_own_conventions():
    v5 = models_for("v5-mobile")
    assert v5.blobs == ("in0", "out0")
    assert v5.dictionary_includes_blank is False


def test_every_catalogued_model_set_is_installed():
    for name in CATALOG:
        models_for(name).validated()


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
