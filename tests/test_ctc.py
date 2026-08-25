"""CTC greedy decode against hand-built logit vectors."""

import numpy as np
import pytest

from vulkanocr.recognition import CtcDictionaryMismatchError, decode_ctc

# classes: 0 blank, 1 -> "a", 2 -> "b", 3 -> "c"
CHARS = ("a", "b", "c")


def logits(*steps, classes=4):
    out = np.full((len(steps), classes), -10.0, dtype=np.float32)
    for row, (index, score) in enumerate(steps):
        out[row, index] = score
    return out


def test_repeats_collapse_and_blanks_separate():
    #  a a blank a  ->  "aa"
    text, _ = decode_ctc(logits((1, 5), (1, 5), (0, 5), (1, 5)), CHARS)
    assert text == "aa"


def test_blanks_only_is_empty_with_zero_confidence():
    text, confidence = decode_ctc(logits((0, 9), (0, 9)), CHARS)
    assert text == ""
    assert confidence == 0.0


def test_confidence_is_the_mean_of_kept_steps():
    text, confidence = decode_ctc(logits((1, 2), (0, 9), (2, 4)), CHARS)
    assert text == "ab"
    assert confidence == 3.0


def test_out_of_dictionary_classes_are_a_model_mismatch():
    wide = np.full((1, 6), -10.0, dtype=np.float32)
    wide[0, 5] = 8.0  # class 5 -> characters[4], beyond the 3-entry dict
    with pytest.raises(CtcDictionaryMismatchError, match="6 classes.*expects 4"):
        decode_ctc(wide, CHARS)


def test_sequence_abc_decodes_in_order():
    text, _ = decode_ctc(logits((1, 5), (0, 5), (2, 5), (0, 5), (3, 5)), CHARS)
    assert text == "abc"


def test_offset_zero_reads_dictionaries_that_carry_the_blank():
    # Avafly keys files list the CTC blank as their first entry.
    with_blank = ("", "a", "b", "c")
    text, _ = decode_ctc(logits((1, 5), (0, 5), (2, 5)), with_blank, offset=0)
    assert text == "ab"


def test_the_wrong_offset_shifts_every_character():
    """The failure mode is a systematic shift, so the test asserts a shift.

    The old assertion decoded two classes and checked one letter — it passed
    for truncation as readily as for the shift it is named after. Three
    non-adjacent classes make the displacement visible end to end.
    """
    characters = ("", "a", "b", "c", "d", "e")
    steps = logits((1, 5), (3, 5), (5, 5), classes=6)

    right, _ = decode_ctc(steps, characters, offset=0)
    assert right == "ace"
    with pytest.raises(CtcDictionaryMismatchError, match="expects 7"):
        decode_ctc(steps, characters, offset=1)
