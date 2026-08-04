"""Image -> PDF conversion and paper-size detection/mapping.

Paper sizes are expressed in points (1/72 inch), the unit reportlab uses.
Images combined into a PDF by this bot always use "Short" (see
images_to_pdf callers) -- only genuine document/PDF attachments go through
size detection via detect_pdf_paper_size / the AI classifier.
"""

from __future__ import annotations

import os

from PIL import Image
from pypdf import PdfReader
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

# Name -> (width_pt, height_pt), all in portrait orientation.
# "Short bond paper" = Letter (8.5 x 11 in), "long bond paper" = Legal
# (8.5 x 14 in) -- these are the two options the bot is allowed to choose
# between.
PAPER_SIZES: dict[str, tuple[float, float]] = {
    "Short": (8.5 * inch, 11 * inch),
    "Long": (8.5 * inch, 14 * inch),
}

# Human-friendly labels shown in Discord messages / logs.
PAPER_SIZE_LABELS: dict[str, str] = {
    "Short": "Short bond paper (8.5\" x 11\")",
    "Long": "Long bond paper (8.5\" x 14\")",
}


def resolve_paper_size(name: str | None, default: str) -> tuple[str, tuple[float, float]]:
    """Returns (name, (width, height)) for a paper size name, falling back
    to `default` if `name` is None or not recognized."""
    if name and name in PAPER_SIZES:
        return name, PAPER_SIZES[name]
    return default, PAPER_SIZES[default]


def images_to_pdf(image_paths: list[str], output_path: str, paper_size_pt: tuple[float, float]) -> str:
    """Places each image on its own page, scaled to fit the page as large
    as possible while preserving aspect ratio, and centered."""
    page_width, page_height = paper_size_pt
    c = canvas.Canvas(output_path, pagesize=paper_size_pt)

    for path in image_paths:
        with Image.open(path) as img:
            # Respect EXIF orientation so photos from phones print upright.
            img = _apply_exif_orientation(img)
            img_w, img_h = img.size

            scale = min(page_width / img_w, page_height / img_h)
            draw_w, draw_h = img_w * scale, img_h * scale
            x = (page_width - draw_w) / 2
            y = (page_height - draw_h) / 2

            c.drawImage(
                path, x, y, width=draw_w, height=draw_h,
                preserveAspectRatio=True, anchor="c",
            )
        c.showPage()

    c.save()
    return output_path


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    try:
        from PIL import ImageOps
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def detect_pdf_paper_size(pdf_path: str, supported_sizes: list[str], default: str) -> str:
    """For an existing PDF attachment, reads the first page's dimensions
    and maps them to the closest supported paper size name."""
    try:
        reader = PdfReader(pdf_path)
        box = reader.pages[0].mediabox
        w, h = float(box.width), float(box.height)
    except Exception:
        return default

    best_name = default
    best_diff = float("inf")
    for name in supported_sizes:
        if name not in PAPER_SIZES:
            continue
        pw, ph = PAPER_SIZES[name]
        diff = abs(pw - w) + abs(ph - h)
        diff_rotated = abs(pw - h) + abs(ph - w)
        diff = min(diff, diff_rotated)
        if diff < best_diff:
            best_diff = diff
            best_name = name

    return best_name


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


def is_pdf_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".pdf"
