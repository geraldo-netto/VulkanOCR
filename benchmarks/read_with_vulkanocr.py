"""Read the corpus with the ncnn/Vulkan spike engine and score it."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from corpusrun import run_corpus
from scoring import load_rgb

from vulkanocr import PRECISIONS, options_for_precision
from vulkanocr.catalog import CATALOG, models_for
from vulkanocr.engine import OcrEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=pathlib.Path)
    parser.add_argument("model_set", choices=sorted(CATALOG))
    parser.add_argument("--precision", choices=PRECISIONS, default="fp32")
    arguments = parser.parse_args()
    corpus = arguments.corpus
    model_set = arguments.model_set
    precision = arguments.precision
    cases = json.loads((corpus / "ground-truth.json").read_text(encoding="utf-8"))

    with OcrEngine(models_for(model_set), options=options_for_precision(precision)) as engine:

        def read(path) -> list[str]:
            # read() already sorts by (center_y, center_x); sorting again here
            # implied the engine\'s order could not be trusted.
            return [line.text for line in engine.read(load_rgb(path)).lines]

        suffix = f"-{precision}" if precision != "fp32" else ""
        tag_suffix = f"+{precision}" if precision != "fp32" else ""
        run_corpus(
            corpus,
            cases,
            read,
            tag=f"vulkanocr/{model_set}{tag_suffix}",
            device=engine.device_name,
            out=corpus.parent / f"results-vulkanocr-{model_set}{suffix}.json",
        )
        print("device:", engine.device_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
