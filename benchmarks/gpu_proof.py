"""Per-process GPU proof: amdgpu fdinfo counters for this PID only."""

import os
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from scoring import load_rgb

from vulkanocr.catalog import models_for
from vulkanocr.engine import OcrEngine


def drm_counters():
    """Nanoseconds of GPU engine time and VRAM this process holds."""
    totals = {}
    for fd in pathlib.Path(f"/proc/{os.getpid()}/fdinfo").iterdir():
        try:
            text = fd.read_text()
        except OSError:
            continue
        if "drm-driver" not in text:
            continue
        for key, value in re.findall(r"^(drm-(?:engine|memory)-\w+):\s+(\d+)", text, re.M):
            totals[key] = totals.get(key, 0) + int(value)
    return totals


rgb = load_rgb(sys.argv[1])
engine = OcrEngine(models_for("v6-medium"))
engine.read(rgb)
print("device:", engine.device_name)
before = drm_counters()
start = time.perf_counter()
for _ in range(5):
    engine.read(rgb)
wall = (time.perf_counter() - start) / 5
after = drm_counters()
for key in sorted(set(before) | set(after)):
    delta = after.get(key, 0) - before.get(key, 0)
    if key.startswith("drm-engine"):
        print(f"  {key:24} +{delta / 1e6 / 5:9.1f} ms of GPU time per read")
    else:
        print(f"  {key:24}  {after.get(key, 0) / 1024:9.1f} MiB held")
print(f"  wall per read            {wall * 1000:9.1f} ms")
