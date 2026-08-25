"""Packing crops side by side, shared by the strip experiments.

Two scripts carried this body verbatim; a third variant differed only in
fusing the decode. One copy, so the gap constant and the span arithmetic
cannot drift apart between experiments that exist to be compared.
"""

from __future__ import annotations

import numpy as np

# White columns between crops. Wider than the recognition net's receptive
# field is supposed to reach — the one-strip experiment measures whether it
# actually holds.
GAP = 32


def pack(crops: list[np.ndarray], gap: int = GAP) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """One 48-high strip holding every crop, and where each one landed."""
    if not crops:
        raise ValueError("cannot pack an empty crop sequence")
    width = sum(crop.shape[1] for crop in crops) + gap * (len(crops) - 1)
    strip = np.full((48, width, 3), 255, dtype=np.uint8)
    spans, x = [], 0
    for crop in crops:
        strip[:, x : x + crop.shape[1]] = crop
        spans.append((x, x + crop.shape[1]))
        x += crop.shape[1] + gap
    return strip, spans


def decode_spans(engine, strip: np.ndarray, spans: list[tuple[int, int]]) -> list[str]:
    """Decode each packed crop's segment of the strip's CTC output."""
    logits = engine.logits(strip)
    scale = logits.shape[0] / strip.shape[1]
    texts = []
    for left, right in spans:
        lo = int(round(left * scale))
        hi = max(int(round(right * scale)), lo + 1)
        texts.append(engine.decode(logits[lo:hi])[0])
    return texts
