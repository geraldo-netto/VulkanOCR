"""Render a ground-truth OCR corpus: known text, known degradations.

Synthetic on purpose. A comparison needs identical inputs and exact ground
truth for every engine; a hand-transcribed scan corpus is the honest ideal and
is days of work, so this trades realism for exactness and says so. The
degradations are the ones that actually separate OCR engines in practice:
size, font, contrast, blur, noise, JPEG, skew, and dense UI text.
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np
from corpus_schema import SCHEMA_VERSION, write_manifest
from PIL import Image, ImageDraw, ImageFont

FONTS = {
    "sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
}
FONT_FACTS = {
    "sans": ("DejaVu Sans", "DejaVuSans.ttf"),
    "serif": ("DejaVu Serif", "DejaVuSerif.ttf"),
    "mono": ("DejaVu Sans Mono", "DejaVuSansMono.ttf"),
}
FONT_LICENSE = "Bitstream Vera Fonts Copyright"

DOCUMENTS = [
    {
        "script": "Latn",
        "language": "en",
        "lines": [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump!",
        ],
    },
    {
        "script": "Latn",
        "language": "en",
        "lines": [
            "Invoice 2026-08-441",
            "Subtotal: 1,284.50 EUR",
            "VAT (23%): 295.44 EUR",
            "Total due: 1,579.94 EUR",
        ],
    },
    {
        "script": "Latn",
        "language": "pt",
        "lines": [
            "O gato subiu no telhado e ficou",
            "olhando a lua durante a noite fria.",
            "Ação, coração, informação.",
        ],
    },
    {
        "script": "Latn",
        "language": "en",
        "lines": [
            "def read(self, rgb: np.ndarray) -> OcrResult:",
            "    regions = detect_regions(self._det, rgb)",
            "    return OcrResult(self.device_name, lines)",
        ],
    },
    {
        "script": "Latn",
        "language": "en",
        "lines": [
            "Hardware health 98%",
            "Queue depth: 0 jobs",
            "Radeon RX 6600 XT - gpu 41 C",
            "Media transcription idle",
        ],
    },
]


def render(lines, font_path, size, width=900, pad=24):
    font = ImageFont.truetype(font_path, size)
    height = pad * 2 + int(size * 1.6) * len(lines)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = pad
    for line in lines:
        draw.text((pad, y), line, font=font, fill="black")
        y += int(size * 1.6)
    return np.array(image)


def skew(array, degrees):
    height, width = array.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(
        array,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def blur(array, radius):
    return cv2.GaussianBlur(array, (radius, radius), 0)


def noisy(array, sigma):
    noise = np.random.default_rng(7).normal(0, sigma, array.shape)
    return np.clip(array.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg(array, quality):
    """Round-trip through JPEG, so the corpus carries real compression noise."""
    encoded, buffer = cv2.imencode(".jpg", array[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not encoded:
        raise RuntimeError("the JPEG encoder refused this image")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("the JPEG decoder refused what the encoder produced")
    return decoded[:, :, ::-1]


def faded(array, factor):
    return np.clip(255 - (255 - array.astype(np.float32)) * factor, 0, 255).astype(np.uint8)


def main(output: pathlib.Path | None = None) -> int:
    output = output or pathlib.Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)

    cases = []
    for index, document in enumerate(DOCUMENTS):
        lines = document["lines"]
        base = render(lines, FONTS["sans"], 28)
        variants = {
            "clean-28px-sans": (base, "sans", 28),
            "clean-16px-sans": (render(lines, FONTS["sans"], 16), "sans", 16),
            "clean-12px-sans": (render(lines, FONTS["sans"], 12, width=700), "sans", 12),
            "clean-28px-serif": (render(lines, FONTS["serif"], 28), "serif", 28),
            "clean-28px-mono": (render(lines, FONTS["mono"], 28), "mono", 28),
            "skew-5deg": (skew(base, 5), "sans", 28),
            "skew-12deg": (skew(base, 12), "sans", 28),
            "blur-5px": (blur(base, 5), "sans", 28),
            "noise-sigma25": (noisy(base, 25), "sans", 28),
            "jpeg-q30": (jpeg(base, 30), "sans", 28),
            "faded-40pc": (faded(base, 0.4), "sans", 28),
        }
        for name, (array, font_key, font_px) in variants.items():
            stem = f"case{index:02d}-{name}"
            image = f"{stem}.png"
            cv2.imwrite(str(output / image), array[:, :, ::-1])
            family, font_file = FONT_FACTS[font_key]
            height, width = array.shape[:2]
            cases.append(
                {
                    "id": stem,
                    "script": document["script"],
                    "language": document["language"],
                    "direction": "ltr",
                    "lines": lines,
                    "font": {
                        "family": family,
                        "file": font_file,
                        "license": FONT_LICENSE,
                    },
                    "palette": {"foreground": "#000000", "background": "#FFFFFF"},
                    "size": {"font_px": font_px, "width_px": width, "height_px": height},
                    "background_objects": [],
                    "variant": name,
                    "image": image,
                }
            )

    write_manifest(
        output / "ground-truth.json",
        {"schema_version": SCHEMA_VERSION, "cases": cases},
    )
    print(f"{len(cases)} images, {len(DOCUMENTS)} texts x 11 variants -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
