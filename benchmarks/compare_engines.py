"""Aggregate every engine's results into one comparison."""

from __future__ import annotations

import json
import pathlib
import statistics
import sys


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    files = sorted(root.glob("results-*.json"))
    print(
        f"{'engine':34} {'CER':>7} {'WER':>7} {'exact':>7} "
        f"{'ms p50':>8} {'ms p95':>8} {'total s':>8}"
    )
    print("-" * 86)
    summaries = []
    for path in files:
        doc = json.loads(path.read_text())
        rows = doc["rows"]
        cer = sum(r["char_distance"] for r in rows) / sum(r["char_length"] for r in rows)
        wer = sum(r["word_distance"] for r in rows) / sum(r["word_length"] for r in rows)
        exact = sum(1 for r in rows if r["exact"]) / len(rows)
        times = sorted(r["ms"] for r in rows)
        p50 = statistics.median(times)
        p95 = times[int(len(times) * 0.95) - 1]
        total = sum(times) / 1000
        summaries.append((doc["engine"], doc["device"], cer, wer, exact, p50, p95, total, rows))
        print(
            f"{doc['engine']:34} {cer:7.4f} {wer:7.4f} {exact:6.0%} "
            f"{p50:8.1f} {p95:8.1f} {total:8.1f}"
        )

    print()
    variants = sorted({r["variant"] for _n, _d, *_rest, rows in summaries for r in rows})
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
