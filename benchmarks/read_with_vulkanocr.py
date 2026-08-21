"""Read the corpus with the ncnn/Vulkan spike engine and score it."""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from corpusrun import run_corpus
from scoring import load_rgb

from vulkanocr.catalog import models_for
from vulkanocr.engine import OcrEngine


def main() -> int:
    corpus = pathlib.Path(sys.argv[1])
    model_set = sys.argv[2]
    fp16 = "--fp16" in sys.argv[3:]
    cases = json.loads((corpus / "ground-truth.json").read_text(encoding="utf-8"))

    with OcrEngine(models_for(model_set), use_fp16=fp16) as engine:

        def read(path) -> str:
            # read() already sorts by (center_y, center_x); sorting again here
            # implied the engine\'s order could not be trusted.
            return " ".join(line.text for line in engine.read(load_rgb(path)).lines)

        suffix = "-fp16" if fp16 else ""
        run_corpus(
            corpus,
            cases,
            read,
            tag=f"vulkanocr/{model_set}{'+fp16' if fp16 else ''}",
            device=engine.device_name,
            out=corpus.parent / f"results-vulkanocr-{model_set}{suffix}.json",
        )
        print("device:", engine.device_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
