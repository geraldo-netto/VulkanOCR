"""How a read is scored against its ground truth.

Character error rate is the standard OCR measure: Levenshtein distance over
characters, divided by the length of the truth. Word error rate is the same
over whitespace-separated tokens. Both are reported per image and aggregated
by total distance over total length — not as a mean of per-image rates, which
would let a two-word image outweigh a paragraph.

Reading order is not a fair thing to demand from a detector, so lines are
aligned by minimum edit cost. Order within each line remains significant.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from pathlib import Path


def load_rgb(path: Path | str):
    """One image as a contiguous RGB array, or a refusal that names the file.

    `cv2.imread` answers `None` for anything it cannot decode — a missing
    path, a truncated PNG, a directory — and every runner here used to
    subscript that answer immediately, turning a wrong path into an opaque
    `TypeError` several frames from the cause.

    cv2 and numpy are imported here rather than at module top so that the
    Tesseract and PaddleOCR runners — which score text and never load an
    image through us — keep working under interpreters without OpenCV.
    """
    import cv2  # noqa: PLC0415 - optional for score-only consumers
    import numpy as np  # noqa: PLC0415

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"cannot read an image from {path}")
    return np.ascontiguousarray(image[:, :, ::-1])


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
    return score_lines([truth], [observed])


def _normalised_lines(lines: Sequence[str]) -> list[str]:
    return [text for line in lines if (text := normalise(line))]


def _assignment_distance(truth: Sequence, observed: Sequence) -> int:
    size = max(len(truth), len(observed))
    if size == 0:
        return 0
    costs = []
    for truth_index in range(size):
        row = []
        for observed_index in range(size):
            if truth_index < len(truth) and observed_index < len(observed):
                cost = levenshtein(truth[truth_index], observed[observed_index])
            elif truth_index < len(truth):
                cost = len(truth[truth_index])
            elif observed_index < len(observed):
                cost = len(observed[observed_index])
            else:
                cost = 0
            row.append(cost)
        costs.append(row)
    return _minimum_assignment_cost(costs)


def _next_assignment_column(
    costs: list[list[int]],
    matched_row: int,
    column: int,
    used: list[bool],
    minimum: list[float],
    path: list[int],
    row_potential: list[float],
    column_potential: list[float],
) -> tuple[float, int]:
    delta = float("inf")
    next_column = 0
    for candidate in range(1, len(used)):
        if used[candidate]:
            continue
        reduced = (
            costs[matched_row - 1][candidate - 1]
            - row_potential[matched_row]
            - column_potential[candidate]
        )
        if reduced < minimum[candidate]:
            minimum[candidate] = reduced
            path[candidate] = column
        if minimum[candidate] < delta:
            delta = minimum[candidate]
            next_column = candidate
    return delta, next_column


def _shift_assignment_potentials(
    delta: float,
    used: list[bool],
    minimum: list[float],
    matching: list[int],
    row_potential: list[float],
    column_potential: list[float],
) -> None:
    for candidate in range(len(used)):
        if used[candidate]:
            row_potential[matching[candidate]] += delta
            column_potential[candidate] -= delta
        else:
            minimum[candidate] -= delta


def _minimum_assignment_cost(costs: list[list[int]]) -> int:
    """Hungarian assignment for a square integer cost matrix, O(lines³)."""
    size = len(costs)
    row_potential = [0.0] * (size + 1)
    column_potential = [0.0] * (size + 1)
    matching = [0] * (size + 1)
    path = [0] * (size + 1)
    for row in range(1, size + 1):
        matching[0] = row
        column = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            delta, next_column = _next_assignment_column(
                costs,
                matching[column],
                column,
                used,
                minimum,
                path,
                row_potential,
                column_potential,
            )
            _shift_assignment_potentials(
                delta, used, minimum, matching, row_potential, column_potential
            )
            column = next_column
            if matching[column] == 0:
                break
        while column:
            previous = path[column]
            matching[column] = matching[previous]
            column = previous
    return int(-column_potential[0])


def score_lines(truth: Sequence[str], observed: Sequence[str]) -> dict:
    """Score line multisets; line traversal order is deliberately excluded."""
    truth_text = _normalised_lines(truth)
    observed_text = _normalised_lines(observed)
    truth_words = [line.split() for line in truth_text]
    observed_words = [line.split() for line in observed_text]
    characters = _assignment_distance(truth_text, observed_text)
    words = _assignment_distance(truth_words, observed_words)
    character_length = sum(map(len, truth_text))
    word_length = sum(map(len, truth_words))
    return {
        "char_distance": characters,
        "char_length": character_length,
        "word_distance": words,
        "word_length": word_length,
        "cer": characters / max(1, character_length),
        "wer": words / max(1, word_length),
        "exact": sorted(truth_text) == sorted(observed_text),
    }
