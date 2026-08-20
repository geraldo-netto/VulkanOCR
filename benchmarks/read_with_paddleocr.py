"""Read the corpus with upstream PaddleOCR (CPU) and score it identically.

Run from PaddleOCR's own virtualenv, not this project's: it needs
`paddlepaddle<3.3` (see `docs/benchmarks.md`) and is deliberately not a
dependency of the engine. That is why the import is unresolvable here.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("FLAGS_call_stack_level", "0")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from paddleocr import PaddleOCR
from scoring import score


def main() -> int:
    corpus = pathlib.Path(sys.argv[1])
    cases = json.loads((corpus / "ground-truth.json").read_text(encoding="utf-8"))
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        enable_mkldnn=False,
    )

    def read(path):
        result = ocr.predict(str(path))
        texts = []
        for page in result:
            data = page.json["res"] if hasattr(page, "json") else page
            texts.extend(data.get("rec_texts", []))
        return " ".join(texts)

    read(corpus / cases[0]["image"])
    rows = []
    for case in cases:
        start = time.perf_counter()
        observed = read(corpus / case["image"])
        elapsed = (time.perf_counter() - start) * 1000
        row = {
            "id": case["id"],
            "variant": case["variant"],
            "ms": elapsed,
            "observed": observed,
            **score(" ".join(case["lines"]), observed),
        }
        rows.append(row)
        print(
            f"{case['id']:34} cer={row['cer']:.3f} wer={row['wer']:.3f} {elapsed:7.1f} ms",
            flush=True,
        )
    out = corpus.parent / "results-paddleocr-cpu.json"
    out.write_text(
        json.dumps(
            {"engine": "paddleocr-3.7.0/PP-OCRv5-mobile", "device": "CPU", "rows": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
