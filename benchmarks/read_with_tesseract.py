"""Read the corpus with Tesseract and score it identically."""

from __future__ import annotations

import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from corpus_schema import load_cases, require_engine_selections
from corpusrun import run_corpus


def main() -> int:
    corpus = pathlib.Path(sys.argv[1])
    psm = sys.argv[2] if len(sys.argv) > 2 else "6"
    cases = load_cases(corpus / "ground-truth.json")
    require_engine_selections(cases, "tesseract")

    def read(path, case) -> list[str]:
        language = case["recognition"]["tesseract"]["language"]
        done = subprocess.run(
            ["tesseract", str(path), "stdout", "-l", language, "--psm", psm],
            capture_output=True,
            text=True,
            check=False,
        )
        return [line for line in done.stdout.splitlines() if line.strip()]

    run_corpus(
        corpus,
        cases,
        read,
        tag=f"tesseract-5.3.4/declared-languages/psm{psm}",
        device="CPU",
        out=corpus.parent / f"results-tesseract-psm{psm}.json",
        selector=lambda case: case["recognition"]["tesseract"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
