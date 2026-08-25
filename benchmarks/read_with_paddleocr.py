"""Read the corpus with upstream PaddleOCR (CPU) and score it identically.

Run from PaddleOCR's own virtualenv, not this project's: install the `paddle`
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

import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("FLAGS_call_stack_level", "0")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from corpus_schema import load_cases, require_engine_selections
from corpusrun import run_corpus
from paddleocr import PaddleOCR


def main() -> int:
    corpus = pathlib.Path(sys.argv[1])
    mkldnn = "--no-mkldnn" not in sys.argv[2:]
    cases = load_cases(corpus / "ground-truth.json")
    selections = require_engine_selections(cases, "paddleocr")
    keys = sorted({(selection["language"], selection["model"]) for selection in selections})
    engines = {
        key: PaddleOCR(
            lang=key[0],
            ocr_version=key[1],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=mkldnn,
        )
        for key in keys
    }
    import paddle

    models = "+".join(
        sorted({ocr._params.get("text_recognition_model_name", "?") for ocr in engines.values()})
    )
    tag = (
        f"paddleocr-{__import__('paddleocr').__version__}/"
        f"paddle-{paddle.__version__}/{models}/onednn-{'on' if mkldnn else 'off'}"
    )

    def read(path, case) -> list[str]:
        selection = case["recognition"]["paddleocr"]
        ocr = engines[selection["language"], selection["model"]]
        result = ocr.predict(str(path))
        texts = []
        for page in result:
            data = page.json["res"] if hasattr(page, "json") else page
            texts.extend(data.get("rec_texts", []))
        return texts

    run_corpus(
        corpus,
        cases,
        read,
        tag=tag,
        device="CPU",
        out=corpus.parent / f"results-paddleocr-onednn-{'on' if mkldnn else 'off'}.json",
        selector=lambda case: case["recognition"]["paddleocr"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
