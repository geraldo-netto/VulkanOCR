"""The catalogue: default model set and per-port facts."""

from ocr_engine import CATALOG, DEFAULT_MODEL, models_for


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
