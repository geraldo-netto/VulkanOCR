"""Read the corpus with the ncnn/Vulkan spike engine and score it."""

from __future__ import annotations

import argparse
import pathlib
import sys
from contextlib import ExitStack

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from corpus_schema import load_cases, require_engine_selections
from corpusrun import run_corpus
from scoring import load_rgb

from vulkanocr import PRECISIONS, options_for_precision
from vulkanocr.catalog import CATALOG, models_for
from vulkanocr.engine import OcrEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=pathlib.Path)
    parser.add_argument("model_set", nargs="?", choices=sorted(CATALOG))
    parser.add_argument("--precision", choices=PRECISIONS, default="fp32")
    arguments = parser.parse_args()
    corpus = arguments.corpus
    precision = arguments.precision
    cases = load_cases(corpus / "ground-truth.json")
    selections = require_engine_selections(cases, "vulkanocr")
    profiles = sorted({selection["model"] for selection in selections})
    if arguments.model_set is not None and profiles != [arguments.model_set]:
        parser.error(
            f"model override {arguments.model_set!r} differs from declared profiles {profiles}"
        )

    with ExitStack() as stack:
        engines = {
            profile: stack.enter_context(
                OcrEngine(models_for(profile), options=options_for_precision(precision))
            )
            for profile in profiles
        }

        def read(path, case) -> list[str]:
            # read() already sorts by (center_y, center_x); sorting again here
            # implied the engine\'s order could not be trusted.
            profile = case["recognition"]["vulkanocr"]["model"]
            engine = engines[profile]
            return [line.text for line in engine.read(load_rgb(path)).lines]

        suffix = f"-{precision}" if precision != "fp32" else ""
        tag_suffix = f"+{precision}" if precision != "fp32" else ""
        profile_tag = "+".join(profiles)
        devices = sorted({engine.device_name for engine in engines.values()})
        run_corpus(
            corpus,
            cases,
            read,
            tag=f"vulkanocr/{profile_tag}{tag_suffix}",
            device=" + ".join(devices),
            out=corpus.parent / f"results-vulkanocr-{profile_tag}{suffix}.json",
            selector=lambda case: case["recognition"]["vulkanocr"],
        )
        print("device:", " + ".join(devices))
    return 0


if __name__ == "__main__":
    sys.exit(main())
