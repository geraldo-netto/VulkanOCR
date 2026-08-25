"""Read the corpus with Tesseract and score it identically."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from corpusrun import run_corpus


def main() -> int:
    corpus = pathlib.Path(sys.argv[1])
    psm = sys.argv[2] if len(sys.argv) > 2 else "6"
    cases = json.loads((corpus / "ground-truth.json").read_text(encoding="utf-8"))

    def read(path) -> list[str]:
        done = subprocess.run(
            ["tesseract", str(path), "stdout", "-l", "eng", "--psm", psm],
            capture_output=True,
            text=True,
            check=False,
        )
        return [line for line in done.stdout.splitlines() if line.strip()]

    run_corpus(
        corpus,
        cases,
        read,
        tag=f"tesseract-5.3.4/psm{psm}",
        device="CPU",
        out=corpus.parent / f"results-tesseract-psm{psm}.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
