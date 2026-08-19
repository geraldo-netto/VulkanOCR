"""Throwaway PP-OCRv5 spike: det -> boxes -> crops -> rec -> text, on Vulkan.

Ported from nihui/ncnn-android-ppocrv5 (BSD 3-Clause, Tencent) app/src/main/jni/
ppocrv5.cpp. Purpose is a yes/no answer on OMNI-0351, not production code:
no bounds, no cancellation, no lease, no contracts.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import ncnn
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "nihui-port/app/src/main/assets"
KEYS = HERE / "ppocrv5_keys.txt"

DET_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
DET_NORM = (1 / 0.229 / 255, 1 / 0.224 / 255, 1 / 0.225 / 255)
REC_MEAN = (127.5, 127.5, 127.5)
REC_NORM = (1 / 127.5, 1 / 127.5, 1 / 127.5)
BINARY_THRESHOLD = 0.3
BOX_THRESHOLD = 0.6
ENLARGE_RATIO = 1.95
STRIDE = 32
MAX_CANDIDATES = 1000


def hardware_device() -> int:
    """First non-software Vulkan device, honouring the service's no-CPU rule."""
    best = None
    for index in range(ncnn.get_gpu_count()):
        info = ncnn.get_gpu_info(index)
        kind = info.type()
        if kind == 3:  # software rasteriser
            continue
        rank = {0: 0, 1: 1, 2: 2}.get(kind, 3)
        if best is None or rank < best[0]:
            best = (rank, index, info.device_name())
    if best is None:
        raise SystemExit("no hardware Vulkan device; refusing to run on the CPU")
    print(f"device: [{best[1]}] {best[2]}")
    return best[1]


def load_net(param: Path, device: int) -> ncnn.Net:
    net = ncnn.Net()
    net.opt.use_vulkan_compute = True
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.set_vulkan_device(device)
    if net.load_param(str(param)) != 0:
        raise SystemExit(f"cannot load param: {param}")
    if net.load_model(str(param.with_suffix(".bin"))) != 0:
        raise SystemExit(f"cannot load weights for: {param}")
    return net


def detect(net: ncnn.Net, rgb: np.ndarray, target_size: int) -> list[tuple]:
    height, width = rgb.shape[:2]
    scale = 1.0
    w, h = width, height
    if max(w, h) > target_size:
        if w > h:
            scale = target_size / w
            w, h = target_size, int(h * scale)
        else:
            scale = target_size / h
            h, w = target_size, int(w * scale)

    mat = ncnn.Mat.from_pixels_resize(
        np.ascontiguousarray(rgb), ncnn.Mat.PixelType.PIXEL_RGB2BGR, width, height, w, h
    )
    wpad = (w + STRIDE - 1) // STRIDE * STRIDE - w
    hpad = (h + STRIDE - 1) // STRIDE * STRIDE - h
    padded = ncnn.Mat()
    ncnn.copy_make_border(
        mat, padded, hpad // 2, hpad - hpad // 2, wpad // 2, wpad - wpad // 2,
        ncnn.BorderType.BORDER_CONSTANT, 114.0,
    )
    padded.substract_mean_normalize(DET_MEAN, DET_NORM)

    extractor = net.create_extractor()
    extractor.input("in0", padded)
    code, out = extractor.extract("out0")
    if code != 0:
        raise SystemExit("detection failed")
    probability = np.array(out)[0]
    del extractor

    bitmap = (probability > BINARY_THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours[:MAX_CANDIDATES]:
        if len(contour) <= 2:
            continue
        score = contour_score(probability, contour)
        if score < BOX_THRESHOLD:
            continue
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
        if max(rw, rh) < 3 * scale:
            continue

        orientation = 0
        if -30 <= angle <= 30 and rh > rw * 2.7:
            orientation = 1
        if (angle <= -60 or angle >= 60) and rw > rh * 2.7:
            orientation = 1
        if angle < -30:
            angle += 180
        if orientation == 0 and angle < 30:
            angle += 90
            rw, rh = rh, rw
        if orientation == 1 and angle >= 60:
            angle -= 90
            rw, rh = rh, rw

        rh = rh + rw * (ENLARGE_RATIO - 1)
        rw = rw * ENLARGE_RATIO
        cx = (cx - wpad // 2) / scale
        cy = (cy - hpad // 2) / scale
        boxes.append((((cx, cy), (rw / scale, rh / scale), angle), orientation, score))
    return boxes


def contour_score(probability: np.ndarray, contour: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    x, y = max(x, 0), max(y, 0)
    w = min(w, probability.shape[1] - x)
    h = min(h, probability.shape[0] - y)
    if w <= 0 or h <= 0:
        return 0.0
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour - np.array([x, y])], 255)
    return float(cv2.mean(probability[y : y + h, x : x + w], mask=mask)[0])


def crop(rgb: np.ndarray, rrect, orientation: int) -> np.ndarray:
    (_, (rw, rh), _) = rrect
    target_height = 48
    target_width = max(int(rh * target_height / max(rw, 1e-6)), 1)
    corners = cv2.boxPoints(rrect)
    order = (0, 1, 3) if orientation == 0 else (2, 3, 1)
    source = np.float32([corners[index] for index in order])
    destination = np.float32([[0, 0], [target_width, 0], [0, target_height]])
    matrix = cv2.getAffineTransform(source, destination)
    return cv2.warpAffine(
        rgb, matrix, (target_width, target_height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def recognise(net: ncnn.Net, patch: np.ndarray, characters: list[str]) -> tuple[str, float]:
    height, width = patch.shape[:2]
    mat = ncnn.Mat.from_pixels(
        np.ascontiguousarray(patch), ncnn.Mat.PixelType.PIXEL_RGB2BGR, width, height
    )
    mat.substract_mean_normalize(REC_MEAN, REC_NORM)
    extractor = net.create_extractor()
    extractor.input("in0", mat)
    code, out = extractor.extract("out0")
    if code != 0:
        raise SystemExit("recognition failed")
    logits = np.array(out)
    del extractor
    if logits.ndim == 3:
        logits = logits[0]

    indices = logits.argmax(axis=1)
    scores = logits.max(axis=1)
    text: list[str] = []
    confidences: list[float] = []
    last = 0
    for index, score in zip(indices, scores, strict=True):
        index = int(index)
        if index == last:
            continue
        last = index
        if index <= 0:
            continue
        character = index - 1
        if character < len(characters):
            text.append(characters[character])
            confidences.append(float(score))
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    return "".join(text), mean_confidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--models", default="mobile", choices=("mobile", "server"))
    parser.add_argument("--target-size", type=int, default=640)
    arguments = parser.parse_args()

    characters = KEYS.read_text(encoding="utf-8").split("\n")
    image = cv2.imread(arguments.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {arguments.image}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"image: {rgb.shape[1]}x{rgb.shape[0]}  models: {arguments.models}")

    device = hardware_device()
    started = time.monotonic()
    det = load_net(ASSETS / f"PP_OCRv5_{arguments.models}_det.ncnn.param", device)
    rec = load_net(ASSETS / f"PP_OCRv5_{arguments.models}_rec.ncnn.param", device)
    load_ms = (time.monotonic() - started) * 1000

    started = time.monotonic()
    boxes = detect(det, rgb, arguments.target_size)
    detect_ms = (time.monotonic() - started) * 1000

    started = time.monotonic()
    lines = []
    for rrect, orientation, score in boxes:
        patch = crop(rgb, rrect, orientation)
        if patch.size == 0:
            continue
        text, confidence = recognise(rec, patch, characters)
        if text:
            lines.append((rrect[0][1], rrect[0][0], text, score, confidence))
    recognise_ms = (time.monotonic() - started) * 1000

    lines.sort()
    print(f"\nload {load_ms:.0f} ms | detect {detect_ms:.0f} ms | "
          f"recognise {recognise_ms:.0f} ms for {len(boxes)} boxes\n")
    for y, x, text, box_score, confidence in lines:
        print(f"  ({x:5.0f},{y:5.0f}) box={box_score:.2f} conf={confidence:.2f}  {text}")
    print(f"\n{len(lines)} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
