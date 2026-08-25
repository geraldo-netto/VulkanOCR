"""False-positive policy stays contextual instead of suppressing scripts globally."""

import pytest

from vulkanocr.policy import FalsePositivePolicy, RecognitionContext, scripts_in_text


def test_script_detection_covers_catalog_corpus_scripts():
    assert scripts_in_text("Latin Ελληνικά Русский שלום العربية ქართული 花 かな カナ") == {
        "Latn",
        "Grek",
        "Cyrl",
        "Hebr",
        "Arab",
        "Geor",
        "Hani",
        "Hira",
        "Kana",
    }


def test_known_low_confidence_cjk_reading_is_rejected_on_a_latin_page():
    policy = FalsePositivePolicy(frozenset({"花", "回"}), below_confidence=0.6)
    context = RecognitionContext(page_languages=frozenset({"en"}))

    decision = policy.decide("花", 0.41, context)

    assert decision.accepted is False
    assert decision.reason == "known low-confidence reading outside the expected page scripts"


def test_legitimate_cjk_is_retained_from_page_or_model_context():
    policy = FalsePositivePolicy(frozenset({"花", "回"}), below_confidence=0.6)

    assert policy.decide(
        "花", 0.41, RecognitionContext(page_languages=frozenset({"zh"}))
    ).accepted
    assert policy.decide(
        "回", 0.41, RecognitionContext(model_scripts=frozenset({"Hani"}))
    ).accepted


def test_policy_is_permissive_without_context_and_for_other_readings():
    policy = FalsePositivePolicy(frozenset({"花", "回"}), below_confidence=0.6)

    assert policy.decide("花", 0.1).accepted
    assert policy.decide(
        "漢字", 0.1, RecognitionContext(page_languages=frozenset({"en"}))
    ).accepted


def test_confidence_boundary_and_configuration_are_validated():
    policy = FalsePositivePolicy(frozenset({"花"}), below_confidence=0.6)
    latin = RecognitionContext(page_scripts=frozenset({"Latn"}))

    assert policy.decide("花", 0.6, latin).accepted
    with pytest.raises(ValueError, match="between 0 and 1"):
        FalsePositivePolicy(below_confidence=1.1)
