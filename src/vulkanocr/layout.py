"""Conservative whitespace reconstruction from recognised-region geometry."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .engine import OcrLine

ANGLE_TOLERANCE_DEGREES = 12.0
PERPENDICULAR_TOLERANCE = 0.6
MAX_GAP_THICKNESSES = 4.0
MAX_OVERLAP_THICKNESSES = 2.0
MAX_OVERLAP_FRACTION = 0.5
MIN_SPACE_GAP_THICKNESSES = 0.25
SPACE_WIDTH_THICKNESSES = 0.5
MAX_INFERRED_SPACES = 8

_MIN_AXIS_DOT = math.cos(math.radians(ANGLE_TOLERANCE_DEGREES))
_OPENING_PUNCTUATION = frozenset("([{<«‹“‘「『【〔〈《（［｛")
_CLOSING_PUNCTUATION = frozenset(")]}>»›”’」』】〕〉》）］｝,.;:!?%‰،؛؟。．，、：；！？٪")

Direction = Literal["ltr", "rtl", "neutral", "mixed"]
Axis = tuple[float, float]


def reconstruct_whitespace(lines: Iterable[OcrLine]) -> tuple[OcrLine, ...]:
    """Join compatible recognised regions and infer their separators.

    CTC output inside each region is opaque and remains byte-for-byte intact.
    The complete contract, thresholds, and limits live in
    ``docs/whitespace.md``.
    """
    source = tuple(lines)
    if len(source) < 2:
        return source
    components = _visual_line_components(source)
    reconstructed = []
    for component in components:
        reconstructed.extend(_join_component([source[index] for index in component]))
    return tuple(reconstructed)


def _visual_line_components(lines: Sequence[OcrLine]) -> list[list[int]]:
    parents = list(range(len(lines)))
    for left_index, left in enumerate(lines):
        for right_index in range(left_index + 1, len(lines)):
            if _same_visual_line(left, lines[right_index]):
                _union(parents, left_index, right_index)

    grouped: dict[int, list[int]] = {}
    for index in range(len(lines)):
        grouped.setdefault(_find(parents, index), []).append(index)
    return sorted(grouped.values(), key=min)


def _find(parents: list[int], index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def _union(parents: list[int], left: int, right: int) -> None:
    left_root, right_root = _find(parents, left), _find(parents, right)
    if left_root != right_root:
        parents[right_root] = left_root


def _same_visual_line(left: OcrLine, right: OcrLine) -> bool:
    if left.vertical != right.vertical or not _directions_compatible(left.text, right.text):
        return False
    axis = _line_axis(left)
    right_axis = _line_axis(right)
    if _dot(axis, right_axis) < _MIN_AXIS_DOT:
        return False
    dx = right.center_x - left.center_x
    dy = right.center_y - left.center_y
    normal = (-axis[1], axis[0])
    thickness = max(left.thickness, right.thickness)
    if abs(dx * normal[0] + dy * normal[1]) > PERPENDICULAR_TOLERANCE * thickness:
        return False
    centre_distance = abs(dx * axis[0] + dy * axis[1])
    gap = centre_distance - (left.length + right.length) / 2.0
    overlap_limit = min(
        MAX_OVERLAP_THICKNESSES * thickness,
        MAX_OVERLAP_FRACTION * min(left.length, right.length),
    )
    return -overlap_limit <= gap <= MAX_GAP_THICKNESSES * thickness


def _join_component(lines: Sequence[OcrLine]) -> tuple[OcrLine, ...]:
    direction = _component_direction(lines)
    if len(lines) == 1 or direction == "mixed":
        return tuple(lines)
    axis = _component_axis(lines)
    rtl = direction == "rtl"
    ordered = sorted(lines, key=lambda line: _projection(line, axis), reverse=rtl)
    thickness = _median(line.thickness for line in ordered)
    text = ordered[0].text
    for index in range(len(ordered) - 1):
        left, right = ordered[index], ordered[index + 1]
        text += _separator(left, right, axis, thickness) + right.text
    return (_combined_line(ordered, text, axis),)


def _separator(left: OcrLine, right: OcrLine, axis: Axis, thickness: float) -> str:
    if left.text[-1].isspace() or right.text[0].isspace():
        return ""
    if _is_opening(left.text[-1]) or _is_closing(right.text[0]):
        return ""
    left_position = _projection(left, axis)
    right_position = _projection(right, axis)
    gap = abs(right_position - left_position) - (left.length + right.length) / 2.0
    if gap < MIN_SPACE_GAP_THICKNESSES * thickness:
        return ""
    raw_count = math.floor(gap / (SPACE_WIDTH_THICKNESSES * thickness) + 0.5)
    return " " * min(max(raw_count, 1), MAX_INFERRED_SPACES)


def _combined_line(lines: Sequence[OcrLine], text: str, axis: Axis) -> OcrLine:
    normal = (-axis[1], axis[0])
    along = [_projection(line, axis) for line in lines]
    across = [_projection(line, normal) for line in lines]
    along_min = min(value - line.length / 2.0 for value, line in zip(along, lines, strict=True))
    along_max = max(value + line.length / 2.0 for value, line in zip(along, lines, strict=True))
    across_min = min(
        value - line.thickness / 2.0 for value, line in zip(across, lines, strict=True)
    )
    across_max = max(
        value + line.thickness / 2.0 for value, line in zip(across, lines, strict=True)
    )
    along_mid = (along_min + along_max) / 2.0
    across_mid = (across_min + across_max) / 2.0
    return replace(
        lines[0],
        text=text,
        confidence=_weighted(lines, "confidence"),
        box_score=_weighted(lines, "box_score"),
        center_x=axis[0] * along_mid + normal[0] * across_mid,
        center_y=axis[1] * along_mid + normal[1] * across_mid,
        thickness=across_max - across_min,
        length=along_max - along_min,
        angle=_axis_angle(axis),
    )


def _weighted(lines: Sequence[OcrLine], attribute: Literal["confidence", "box_score"]) -> float:
    weights = [sum(not character.isspace() for character in line.text) for line in lines]
    if not any(weights):
        weights = [1] * len(lines)
    return sum(
        getattr(line, attribute) * weight for line, weight in zip(lines, weights, strict=True)
    ) / sum(weights)


def _component_axis(lines: Sequence[OcrLine]) -> Axis:
    axes = [_line_axis(line) for line in lines]
    x = sum(axis[0] for axis in axes)
    y = sum(axis[1] for axis in axes)
    magnitude = math.hypot(x, y)
    return x / magnitude, y / magnitude


def _line_axis(line: OcrLine) -> Axis:
    radians = math.radians(line.angle)
    axis = (math.sin(radians), -math.cos(radians))
    if (line.vertical and axis[1] < 0.0) or (not line.vertical and axis[0] < 0.0):
        return -axis[0], -axis[1]
    return axis


def _axis_angle(axis: Axis) -> float:
    angle = math.degrees(math.atan2(axis[0], -axis[1]))
    if angle > 90.0:
        angle -= 180.0
    if angle <= -90.0:
        angle += 180.0
    return angle


def _projection(line: OcrLine, axis: Axis) -> float:
    return line.center_x * axis[0] + line.center_y * axis[1]


def _directions_compatible(left: str, right: str) -> bool:
    left_direction, right_direction = _text_direction(left), _text_direction(right)
    if "mixed" in (left_direction, right_direction):
        return False
    return (
        left_direction == "neutral"
        or right_direction == "neutral"
        or left_direction == right_direction
    )


def _component_direction(lines: Sequence[OcrLine]) -> Direction:
    directions = {_text_direction(line.text) for line in lines}
    if "mixed" in directions or {"ltr", "rtl"} <= directions:
        return "mixed"
    if directions <= {"rtl", "neutral"} and "rtl" in directions:
        return "rtl"
    if directions <= {"ltr", "neutral"} and "ltr" in directions:
        return "ltr"
    return "neutral"


def _text_direction(text: str) -> Direction:
    directions = {
        "rtl" if unicodedata.bidirectional(character) in {"R", "AL"} else "ltr"
        for character in text
        if unicodedata.bidirectional(character) in {"L", "R", "AL"}
    }
    if len(directions) > 1:
        return "mixed"
    if "rtl" in directions:
        return "rtl"
    if "ltr" in directions:
        return "ltr"
    return "neutral"


def _is_opening(character: str) -> bool:
    return character in _OPENING_PUNCTUATION or unicodedata.category(character) in {"Ps", "Pi"}


def _is_closing(character: str) -> bool:
    return character in _CLOSING_PUNCTUATION or unicodedata.category(character) in {"Pe", "Pf"}


def _dot(left: Axis, right: Axis) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
