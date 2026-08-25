"""Aggregate every engine's results into one comparison."""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys


def _manifest_path(path: pathlib.Path) -> pathlib.Path:
    return path / "ground-truth.json" if path.is_dir() else path


def _expected_cases(path: pathlib.Path) -> dict[str, str]:
    cases = json.loads(_manifest_path(path).read_text(encoding="utf-8"))
    expected = {case["id"]: case["variant"] for case in cases}
    if len(expected) != len(cases):
        raise ValueError("the corpus manifest contains duplicate case ids")
    return expected


def validate_result_rows(engine: str, rows: list[dict], expected: dict[str, str]) -> None:
    ids = [row["id"] for row in rows]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    actual = {row["id"]: row["variant"] for row in rows}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = sorted(
        case_id for case_id in set(expected) & set(actual) if expected[case_id] != actual[case_id]
    )
    problems = []
    if duplicates:
        problems.append(f"duplicate ids {duplicates}")
    if missing:
        problems.append(f"missing ids {missing}")
    if extra:
        problems.append(f"extra ids {extra}")
    if mismatched:
        problems.append(f"wrong variants {mismatched}")
    if problems:
        raise ValueError(f"{engine}: result cases differ from the manifest: {'; '.join(problems)}")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare_engines.py RESULTS_ROOT CORPUS_OR_MANIFEST", file=sys.stderr)
        return 2
    root = pathlib.Path(sys.argv[1])
    try:
        expected = _expected_cases(pathlib.Path(sys.argv[2]))
        documents = []
        for path in sorted(root.glob("results-*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            validate_result_rows(document["engine"], document["rows"], expected)
            documents.append(document)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"comparison refused: {error}", file=sys.stderr)
        return 2
    print(
        f"{'engine':34} {'CER':>7} {'WER':>7} {'exact':>7} "
        f"{'ms p50':>8} {'ms p95':>8} {'total s':>8}"
    )
    print("-" * 86)
    summaries = []
    for doc in documents:
        rows = doc["rows"]
        cer = sum(r["char_distance"] for r in rows) / sum(r["char_length"] for r in rows)
        wer = sum(r["word_distance"] for r in rows) / sum(r["word_length"] for r in rows)
        exact = sum(1 for r in rows if r["exact"]) / len(rows)
        times = sorted(r["ms"] for r in rows)
        p50 = statistics.median(times)
        # The nearest-rank definition: ceil(0.95 * n) as a 1-based rank. The
        # previous expression was one rank low — int(55 * 0.95) - 1 indexed
        # the 52nd of 55 (p92.7), quietly flattering every engine's tail.
        p95 = times[math.ceil(len(times) * 0.95) - 1]
        total = sum(times) / 1000
        summaries.append((doc["engine"], doc["device"], cer, wer, exact, p50, p95, total, rows))
        print(
            f"{doc['engine']:34} {cer:7.4f} {wer:7.4f} {exact:6.0%} "
            f"{p50:8.1f} {p95:8.1f} {total:8.1f}"
        )

    print()
    variants = sorted(set(expected.values()))
    header = f"{'variant':20}" + "".join(
        f"{name.split('/')[-1][:13]:>15}" for name, *_ in summaries
    )
    print(header)
    print("-" * len(header))
    for variant in variants:
        line = f"{variant:20}"
        for *_meta, rows in summaries:
            subset = [r for r in rows if r["variant"] == variant]
            cer = sum(r["char_distance"] for r in subset) / max(
                1, sum(r["char_length"] for r in subset)
            )
            line += f"{cer:15.3f}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
