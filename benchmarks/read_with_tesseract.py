"""Read the corpus with Tesseract and score it identically."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from scoring import score


def main() -> int:
    corpus = pathlib.Path(sys.argv[1])
    psm = sys.argv[2] if len(sys.argv) > 2 else "6"
    cases = json.loads((corpus / "ground-truth.json").read_text())
    rows = []
    subprocess.run(
        ["tesseract", str(corpus / cases[0]["image"]), "stdout", "-l", "eng", "--psm", psm],
        capture_output=True,
        check=False,
    )
    for case in cases:
        start = time.perf_counter()
        done = subprocess.run(
            ["tesseract", str(corpus / case["image"]), "stdout", "-l", "eng", "--psm", psm],
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = (time.perf_counter() - start) * 1000
        observed = done.stdout
        row = {
            "id": case["id"],
            "variant": case["variant"],
            "ms": elapsed,
            "observed": " ".join(observed.split()),
            **score(" ".join(case["lines"]), observed),
        }
        rows.append(row)
        print(
            f"{case['id']:34} cer={row['cer']:.3f} wer={row['wer']:.3f} {elapsed:7.1f} ms",
            flush=True,
        )
    out = corpus.parent / f"results-tesseract-psm{psm}.json"
    out.write_text(
        json.dumps(
            {"engine": f"tesseract-5.3.4/psm{psm}", "device": "CPU", "rows": rows},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
