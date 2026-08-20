"""Where the wall clock goes: Vulkan nets vs the Python/OpenCV glue."""
import pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "mvp"))
import cv2, numpy as np
from ocr_engine import detection, recognition
from ocr_engine.catalog import models_for
from ocr_engine.engine import OcrEngine

rgb = np.ascontiguousarray(cv2.imread(sys.argv[1])[:, :, ::-1])
engine = OcrEngine(models_for(sys.argv[2] if len(sys.argv) > 2 else "v6-medium"))
engine.read(rgb)

detect_ms = crop_ms = rec_ms = 0.0
crops = 0
real_detect, real_crop, real_rec = detection.detect_regions, recognition.crop_region, recognition.recognise_patch

def timed_detect(*a, **k):
    global detect_ms
    start = time.perf_counter(); out = real_detect(*a, **k); detect_ms += time.perf_counter()-start
    return out

def timed_crop(*a, **k):
    global crop_ms
    start = time.perf_counter(); out = real_crop(*a, **k); crop_ms += time.perf_counter()-start
    return out

def timed_rec(*a, **k):
    global rec_ms, crops
    start = time.perf_counter(); out = real_rec(*a, **k); rec_ms += time.perf_counter()-start; crops += 1
    return out

import ocr_engine.engine as engine_module
engine_module.detect_regions, engine_module.crop_region, engine_module.recognise_patch = timed_detect, timed_crop, timed_rec

runs = 5
start = time.perf_counter()
for _ in range(runs):
    result = engine.read(rgb)
wall = time.perf_counter() - start
print(f"lines {len(result.lines)}, crops/read {crops//runs}")
print(f"  detection (1 net call)   {detect_ms/runs*1000:7.1f} ms")
print(f"  cropping  (opencv, cpu)  {crop_ms/runs*1000:7.1f} ms")
print(f"  recognition ({crops//runs} net calls) {rec_ms/runs*1000:7.1f} ms")
print(f"  everything else          {(wall-detect_ms-crop_ms-rec_ms)/runs*1000:7.1f} ms")
print(f"  wall per read            {wall/runs*1000:7.1f} ms")
