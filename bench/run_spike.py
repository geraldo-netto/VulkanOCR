"""Read the corpus with the ncnn/Vulkan spike engine and score it."""
from __future__ import annotations

import json, pathlib, sys, time
sys.path.insert(0, "/backups/disk2/projects/cinnamon/ocr-ncnn-spike/mvp")
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import cv2
import numpy as np
from metrics import score
from ocr_engine.catalog import models_for
from ocr_engine.engine import OcrEngine

corpus = pathlib.Path(sys.argv[1])
model_set = sys.argv[2]
cases = json.loads((corpus / "ground-truth.json").read_text())

engine = OcrEngine(models_for(model_set))
rows = []
# One warm pass first: the first read pays for shader compilation, and a
# comparison of steady-state speed must not charge it to one engine only.
warm = cv2.imread(str(corpus / cases[0]["image"]))[:, :, ::-1]
engine.read(np.ascontiguousarray(warm))

for case in cases:
    rgb = np.ascontiguousarray(cv2.imread(str(corpus / case["image"]))[:, :, ::-1])
    start = time.perf_counter()
    result = engine.read(rgb)
    elapsed = (time.perf_counter() - start) * 1000
    observed = " ".join(line.text for line in sorted(result.lines, key=lambda l: (l.center_y, l.center_x)))
    row = {"id": case["id"], "variant": case["variant"], "ms": elapsed,
           "observed": observed, **score(" ".join(case["lines"]), observed)}
    rows.append(row)
    print(f"{case['id']:34} cer={row['cer']:.3f} wer={row['wer']:.3f} {elapsed:7.1f} ms", flush=True)

out = corpus.parent / f"results-spike-{model_set}.json"
out.write_text(json.dumps({"engine": f"ocr-ncnn-spike/{model_set}", "device": engine.device_name, "rows": rows}, indent=2, ensure_ascii=False))
print("device:", engine.device_name)
