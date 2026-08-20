"""Same models, same image: Vulkan lane against ncnn's own CPU lane."""
import pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "mvp"))
import cv2, numpy as np, ncnn
from ocr_engine.catalog import models_for
from ocr_engine.engine import OcrEngine

rgb = np.ascontiguousarray(cv2.imread(sys.argv[1])[:, :, ::-1])
name = sys.argv[2] if len(sys.argv) > 2 else "v6-medium"

engine = OcrEngine(models_for(name))
engine.read(rgb)
start = time.perf_counter()
for _ in range(3):
    vulkan_result = engine.read(rgb)
vulkan = (time.perf_counter() - start) / 3

# The same engine object with the Vulkan flag off is not possible — ncnn binds
# at load — so a second engine is built whose nets never touch the device.
class CpuEngine(OcrEngine):
    def _load_net(self, param_path):
        net = ncnn.Net()
        net.opt.use_vulkan_compute = False
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        net.load_param(str(param_path))
        net.load_model(str(param_path.with_suffix(".bin")))
        return net

cpu_engine = CpuEngine(models_for(name))
cpu_engine.read(rgb)
start = time.perf_counter()
for _ in range(3):
    cpu_result = cpu_engine.read(rgb)
cpu = (time.perf_counter() - start) / 3

print(f"{name}: vulkan {vulkan*1000:.0f} ms ({len(vulkan_result.lines)} lines) | ncnn cpu {cpu*1000:.0f} ms ({len(cpu_result.lines)} lines)")
