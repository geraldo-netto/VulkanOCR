"""Read the corpus with upstream PaddleOCR (CPU) and score it identically.

Run from PaddleOCR's own virtualenv, not this project's: install the `bench`
extra's pins there (`paddlepaddle>=3.2,<3.3` — 3.3.1's PIR-to-oneDNN
converter refuses every PP-OCR graph, see `docs/benchmarks.md`), which is why
the import is unresolvable here.

oneDNN is on by default because that is upstream's fair fight; pass
`--no-mkldnn` to reproduce the crippled configuration the regression forces
on paddlepaddle 3.3. The engine tag in the result document records the real
versions and the oneDNN state, so a table built from these files cannot claim
a configuration that never ran.
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
    mkldnn = "--no-mkldnn" not in sys.argv[2:]
    cases = json.loads((corpus / "ground-truth.json").read_text(encoding="utf-8"))
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        enable_mkldnn=mkldnn,
    )
    import paddle

    models = ocr._params.get("text_recognition_model_name", "?")
    tag = (
        f"paddleocr-{__import__('paddleocr').__version__}/"
        f"paddle-{paddle.__version__}/{models}/onednn-{'on' if mkldnn else 'off'}"
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
    out = corpus.parent / f"results-paddleocr-onednn-{'on' if mkldnn else 'off'}.json"
    out.write_text(
        json.dumps(
            {"engine": tag, "device": "CPU", "rows": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
