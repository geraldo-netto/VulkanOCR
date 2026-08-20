"""Read the corpus with the ncnn/Vulkan spike engine and score it."""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from scoring import load_rgb, score

from vulkanocr.catalog import models_for
from vulkanocr.engine import OcrEngine


def main() -> int:
    corpus = pathlib.Path(sys.argv[1])
    model_set = sys.argv[2]
    cases = json.loads((corpus / "ground-truth.json").read_text())

    engine = OcrEngine(models_for(model_set))
    rows = []
    # One warm pass first: the first read pays for shader compilation, and a
    # comparison of steady-state speed must not charge it to one engine only.
    engine.read(load_rgb(corpus / cases[0]["image"]))

    for case in cases:
        rgb = load_rgb(corpus / case["image"])
        start = time.perf_counter()
        result = engine.read(rgb)
        elapsed = (time.perf_counter() - start) * 1000
        observed = " ".join(
            line.text
            for line in sorted(result.lines, key=lambda line: (line.center_y, line.center_x))
        )
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

    out = corpus.parent / f"results-spike-{model_set}.json"
    out.write_text(
        json.dumps(
            {"engine": f"vulkanocr/{model_set}", "device": engine.device_name, "rows": rows},
            indent=2,
            ensure_ascii=False,
        )
    )
    print("device:", engine.device_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
