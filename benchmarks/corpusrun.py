"""One measurement loop for every corpus runner.

Three scripts carried this body verbatim — warm pass, per-case timing, the
row document, the per-case print, the results file — and the row schema is
``compare_engines.py``'s input contract, so one runner drifting broke the
table silently. One copy, like ``scoring.py`` and ``strips.py`` before it
(VOCR-0055): a runner supplies ``read(path) -> sequence[str]`` and the tag; the loop
supplies everything the table relies on.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from scoring import score_lines


def run_corpus(corpus: Path, cases: list[dict], read, *, tag: str, device: str, out: Path):
    """Time ``read`` over every case, print progress, and write the document.

    The first image is read once and discarded before timing starts: the
    first read pays for shader compilation (or model download, or a daemon's
    warm-up), and a comparison of steady-state speed must not charge it to
    one engine only.
    """
    read(corpus / cases[0]["image"])
    rows = []
    for case in cases:
        start = time.perf_counter()
        observed_lines = list(read(corpus / case["image"]))
        truth_lines = list(case["lines"])
        elapsed = (time.perf_counter() - start) * 1000
        row = {
            "id": case["id"],
            "variant": case["variant"],
            "ms": elapsed,
            "truth_lines": truth_lines,
            "observed_lines": observed_lines,
            **score_lines(truth_lines, observed_lines),
        }
        rows.append(row)
        print(
            f"{case['id']:34} cer={row['cer']:.3f} wer={row['wer']:.3f} {elapsed:7.1f} ms",
            flush=True,
        )
    out.write_text(
        json.dumps({"engine": tag, "device": device, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows
