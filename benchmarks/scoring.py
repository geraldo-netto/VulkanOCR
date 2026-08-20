"""How a read is scored against its ground truth.

Character error rate is the standard OCR measure: Levenshtein distance over
characters, divided by the length of the truth. Word error rate is the same
over whitespace-separated tokens. Both are reported per image and aggregated
by total distance over total length — not as a mean of per-image rates, which
would let a two-word image outweigh a paragraph.

Reading order is not a fair thing to demand from a detector, so the comparison
normalises both sides to one whitespace-collapsed string per image. That is
the property the caller of an OCR engine actually uses.
"""

from __future__ import annotations

import unicodedata


def normalise(text: str) -> str:
    """One line, one space between tokens, canonical composition."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def levenshtein(left, right) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def score(truth: str, observed: str) -> dict:
    truth_text, observed_text = normalise(truth), normalise(observed)
    characters = levenshtein(truth_text, observed_text)
    words = levenshtein(truth_text.split(), observed_text.split())
    return {
        "char_distance": characters,
        "char_length": len(truth_text),
        "word_distance": words,
        "word_length": len(truth_text.split()),
        "cer": characters / max(1, len(truth_text)),
        "wer": words / max(1, len(truth_text.split())),
        "exact": truth_text == observed_text,
    }
