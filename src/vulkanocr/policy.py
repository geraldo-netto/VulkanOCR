"""Opt-in, context-aware policy for known OCR false positives."""

from __future__ import annotations

from dataclasses import dataclass

_LANGUAGE_SCRIPTS: dict[str, frozenset[str]] = {
    "ar": frozenset({"Arab"}),
    "az": frozenset({"Latn"}),
    "el": frozenset({"Grek"}),
    "en": frozenset({"Latn"}),
    "he": frozenset({"Hebr"}),
    "ja": frozenset({"Hani", "Hira", "Kana"}),
    "ka": frozenset({"Geor"}),
    "pt": frozenset({"Latn"}),
    "ru": frozenset({"Cyrl"}),
    "zh": frozenset({"Hani"}),
}
_SCRIPT_RANGES = (
    (0x0041, 0x024F, "Latn"),
    (0x0370, 0x03FF, "Grek"),
    (0x0400, 0x052F, "Cyrl"),
    (0x0590, 0x05FF, "Hebr"),
    (0x0600, 0x06FF, "Arab"),
    (0x10A0, 0x10FF, "Geor"),
    (0x3040, 0x309F, "Hira"),
    (0x30A0, 0x30FF, "Kana"),
    (0x3400, 0x9FFF, "Hani"),
)


def scripts_in_text(text: str) -> frozenset[str]:
    """Return ISO 15924 scripts carried by letters in one recognition."""
    scripts = set()
    for character in text:
        value = ord(character)
        script = next((name for start, end, name in _SCRIPT_RANGES if start <= value <= end), None)
        if script is not None:
            scripts.add(script)
    return frozenset(scripts)


@dataclass(frozen=True, slots=True)
class RecognitionContext:
    """Expected page content, falling back to model capability when unknown."""

    page_scripts: frozenset[str] = frozenset()
    page_languages: frozenset[str] = frozenset()
    model_scripts: frozenset[str] = frozenset()
    model_languages: frozenset[str] = frozenset()

    @property
    def expected_scripts(self) -> frozenset[str]:
        page = self.page_scripts | _scripts_for_languages(self.page_languages)
        if page:
            return page
        return self.model_scripts | _scripts_for_languages(self.model_languages)


@dataclass(frozen=True, slots=True)
class FalsePositiveDecision:
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class FalsePositivePolicy:
    """Reject configured low-confidence readings only outside expected scripts.

    Empty ``readings`` is permissive. Page context takes precedence over broad
    model capability, and no context also stays permissive. This prevents a
    Latin-page icon guard from becoming a global CJK ban.
    """

    readings: frozenset[str] = frozenset()
    below_confidence: float = 0.6

    def __post_init__(self) -> None:
        if not 0.0 <= self.below_confidence <= 1.0:
            raise ValueError("below_confidence must be between 0 and 1")
        if any(not reading for reading in self.readings):
            raise ValueError("false-positive readings cannot be empty")

    def decide(
        self,
        text: str,
        confidence: float,
        context: RecognitionContext | None = None,
    ) -> FalsePositiveDecision:
        if text not in self.readings or confidence >= self.below_confidence:
            return FalsePositiveDecision(True)
        recognised_scripts = scripts_in_text(text)
        expected_scripts = context.expected_scripts if context is not None else frozenset()
        if not recognised_scripts or not expected_scripts or recognised_scripts & expected_scripts:
            return FalsePositiveDecision(True)
        return FalsePositiveDecision(
            False,
            "known low-confidence reading outside the expected page scripts",
        )


def _scripts_for_languages(languages: frozenset[str]) -> frozenset[str]:
    scripts = set()
    for language in languages:
        base = language.casefold().split("-", 1)[0]
        scripts.update(_LANGUAGE_SCRIPTS.get(base, ()))
    return frozenset(scripts)


__all__ = [
    "FalsePositiveDecision",
    "FalsePositivePolicy",
    "RecognitionContext",
    "scripts_in_text",
]
