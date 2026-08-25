"""One measurement loop for every corpus runner.

Three scripts carried this body verbatim — warm pass, per-case timing, the
row document, the per-case print, the results file — and the row schema is
``compare_engines.py``'s input contract, so one runner drifting broke the
table silently. One copy, like ``scoring.py`` and ``strips.py`` before it
(VOCR-0055): a runner supplies ``read(path, case) -> sequence[str]`` and the
tag; the loop supplies everything the table relies on.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from scoring import score_lines


def validate_case_id_parity(cases: list[dict], rows: list[dict]) -> None:
    expected = Counter(case["id"] for case in cases)
    actual = Counter(row["id"] for row in rows)
    if expected != actual:
        missing = sorted((expected - actual).elements())
        extra = sorted((actual - expected).elements())
        raise ValueError(f"result case ids differ from manifest: missing {missing}; extra {extra}")


def run_corpus(
    corpus: Path,
    cases: list[dict],
    read,
    *,
    tag: str,
    device: str,
    out: Path,
    selector=lambda _case: None,
):
    """Time ``read`` over every case, print progress, and write the document.

    The first image for each declared model/language selection is read once
    and discarded before timing starts, so compilation/model warm-up is not
    charged to one engine only.
    """
    if not cases:
        raise ValueError("a corpus run requires at least one case")
    warmed = set()
    for case in cases:
        selection = selector(case)
        key = json.dumps(selection, sort_keys=True)
        if key not in warmed:
            read(corpus / case["image"], case)
            warmed.add(key)
    rows = []
    for case in cases:
        start = time.perf_counter()
        observed_lines = list(read(corpus / case["image"], case))
        truth_lines = list(case["lines"])
        elapsed = (time.perf_counter() - start) * 1000
        row = {
            "id": case["id"],
            "variant": case["variant"],
            "ms": elapsed,
            "truth_lines": truth_lines,
            "observed_lines": observed_lines,
            "selection": selector(case),
            **score_lines(truth_lines, observed_lines),
        }
        rows.append(row)
        print(
            f"{case['id']:34} cer={row['cer']:.3f} wer={row['wer']:.3f} {elapsed:7.1f} ms",
            flush=True,
        )
    validate_case_id_parity(cases, rows)
    out.write_text(
        json.dumps({"engine": tag, "device": device, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows
