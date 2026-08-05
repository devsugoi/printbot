"""Image -> PDF conversion and paper-size detection/mapping.

Paper sizes are expressed in points (1/72 inch), the unit reportlab uses.
Images combined into a PDF by this bot always use "Short" (see
images_to_pdf callers) -- only genuine document/PDF attachments go through
size detection via detect_pdf_paper_size / the AI classifier.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

if TYPE_CHECKING:
    from .config import OfficeConversionConfig

BACKEND_LIBREOFFICE = "libreoffice"
BACKEND_ASPOSE = "aspose"
BACKEND_CLOUDMERSIVE = "cloudmersive"
_CLOUD_BACKENDS = {BACKEND_ASPOSE, BACKEND_CLOUDMERSIVE}

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
                ImageReader(img), x, y, width=draw_w, height=draw_h,
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


def scaled_pdf_path(original_path: str, target_size: str) -> str:
    """Returns a cache path for a PDF scaled to fit `target_size`."""
    base, ext = os.path.splitext(original_path)
    return f"{base}.{target_size.lower()}_fit{ext}"


def scale_pdf_to_paper_size(
    input_path: str, output_path: str, target_size: str
) -> str:
    """Scales each page of a PDF to fit within the target paper size,
    preserving aspect ratio and centering content on a new page of that
    size."""
    if target_size not in PAPER_SIZES:
        raise ValueError(f"Unsupported target paper size: {target_size}")

    target_w, target_h = PAPER_SIZES[target_size]
    reader = PdfReader(input_path)
    writer = PdfWriter()

    for page in reader.pages:
        page_w = float(page.mediabox.width)
        page_h = float(page.mediabox.height)

        scale = min(target_w / page_w, target_h / page_h)
        scaled_w = page_w * scale
        scaled_h = page_h * scale
        tx = (target_w - scaled_w) / 2
        ty = (target_h - scaled_h) / 2

        page.add_transformation(
            Transformation().scale(sx=scale, sy=scale).translate(tx=tx, ty=ty)
        )
        page.mediabox.lower_left = (0, 0)
        page.mediabox.upper_right = (target_w, target_h)
        writer.add_page(page)

    with open(output_path, "wb") as out:
        writer.write(out)
    return output_path


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


# Office / rich-document formats that CUPS can't print directly but
# LibreOffice can convert to PDF.
OFFICE_EXTENSIONS = {
    ".doc", ".docx", ".odt", ".rtf",
    ".xls", ".xlsx", ".ods",
    ".ppt", ".pptx", ".odp",
}


class OfficeConversionError(RuntimeError):
    """Raised when an office document could not be converted to PDF."""


def is_office_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in OFFICE_EXTENSIONS


def converted_pdf_path(input_path: str) -> str:
    """LibreOffice output path (basename with .pdf extension)."""
    base, _ = os.path.splitext(input_path)
    return base + ".pdf"


def converted_pdf_path_for_backend(input_path: str, backend: str) -> str:
    """Output PDF path for a given conversion backend."""
    base, _ = os.path.splitext(input_path)
    if backend == BACKEND_LIBREOFFICE:
        return base + ".pdf"
    if backend == BACKEND_ASPOSE:
        return base + ".aspose.pdf"
    if backend == BACKEND_CLOUDMERSIVE:
        return base + ".cloudmersive.pdf"
    raise ValueError(f"Unknown office conversion backend: {backend}")


_LIBREOFFICE_PROFILE_DIR = ".libreoffice-printbot"
_REGISTRY_TEMPLATE = Path(__file__).resolve().parent / "libreoffice" / "registrymodifications.xcu"
# Explicit Writer PDF export filter with font embedding (LibreOffice 7.3+).
_WRITER_PDF_EXPORT = (
    'pdf:writer_pdf_Export:{"EmbedStandardFonts":{"type":"boolean","value":"true"}}'
)


def _libreoffice_profile_root() -> Path:
    return Path.cwd() / _LIBREOFFICE_PROFILE_DIR


def _libreoffice_user_installation_uri() -> str:
    return _libreoffice_profile_root().resolve().as_uri()


def _ensure_libreoffice_profile() -> None:
    """Seed an isolated LibreOffice user profile for reproducible headless
    conversions (fonts/layout settings without touching a GUI profile)."""
    user_dir = _libreoffice_profile_root() / "user"
    user_dir.mkdir(parents=True, exist_ok=True)
    registry = user_dir / "registrymodifications.xcu"
    if not registry.exists():
        shutil.copy(_REGISTRY_TEMPLATE, registry)


def _office_pdf_output_path(
    input_path: str, output_dir: str, backend: str = BACKEND_LIBREOFFICE,
) -> str:
    return os.path.join(
        output_dir,
        os.path.basename(converted_pdf_path_for_backend(input_path, backend)),
    )


def office_pdf_cache_valid(
    input_path: str,
    output_dir: str | None = None,
    backend: str = BACKEND_LIBREOFFICE,
) -> bool:
    """True if a converted PDF exists and is at least as new as the source."""
    output_dir = output_dir or os.path.dirname(input_path) or "."
    expected = _office_pdf_output_path(input_path, output_dir, backend)
    if not os.path.exists(expected) or not os.path.exists(input_path):
        return False
    return os.path.getmtime(expected) >= os.path.getmtime(input_path)


def office_to_pdf(
    input_path: str,
    output_dir: str | None = None,
    backend: str = BACKEND_LIBREOFFICE,
    office_config: OfficeConversionConfig | None = None,
    force: bool = False,
) -> str:
    """Converts an office document to PDF using the requested backend.

    Raises OfficeConversionError if conversion fails.
    """
    output_dir = output_dir or os.path.dirname(input_path) or "."
    expected = _office_pdf_output_path(input_path, output_dir, backend)
    if not force and office_pdf_cache_valid(input_path, output_dir, backend):
        return expected

    if backend == BACKEND_LIBREOFFICE:
        return _office_to_pdf_libreoffice(input_path, output_dir, expected)
    if backend == BACKEND_ASPOSE:
        if office_config is None or not office_config.aspose.is_available():
            raise OfficeConversionError(
                "Aspose cloud conversion is not configured. "
                "See README for setup."
            )
        return _office_to_pdf_aspose(
            input_path, expected, office_config.aspose.client_id,
            office_config.aspose.client_secret,
        )
    if backend == BACKEND_CLOUDMERSIVE:
        if office_config is None or not office_config.cloudmersive.is_available():
            raise OfficeConversionError(
                "Cloudmersive cloud conversion is not configured. "
                "See README for setup."
            )
        return _office_to_pdf_cloudmersive(
            input_path, expected, office_config.cloudmersive.api_key,
        )
    raise OfficeConversionError(f"Unknown conversion backend: {backend}")


def _office_to_pdf_libreoffice(
    input_path: str, output_dir: str, expected: str,
) -> str:
    binary = shutil.which("soffice") or shutil.which("libreoffice")
    if binary is None:
        raise OfficeConversionError(
            f"Can't print {os.path.basename(input_path)}: LibreOffice is "
            f"not installed, so it can't be converted to PDF. Install it "
            f"with: sudo apt install -y libreoffice"
        )

    _ensure_libreoffice_profile()
    command = [
        binary, "--headless",
        f"-env:UserInstallation={_libreoffice_user_installation_uri()}",
        "--convert-to", _WRITER_PDF_EXPORT,
        "--outdir", output_dir,
        input_path,
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        raise OfficeConversionError(
            f"Converting {os.path.basename(input_path)} to PDF timed out."
        )
    if result.returncode != 0 or not os.path.exists(expected):
        detail = (result.stderr or result.stdout or "").strip()
        raise OfficeConversionError(
            f"Converting {os.path.basename(input_path)} to PDF failed"
            + (f": {detail}" if detail else ".")
        )
    return expected


def _office_to_pdf_aspose(
    input_path: str, expected: str, client_id: str, client_secret: str,
) -> str:
    try:
        import asposewordscloud
        from asposewordscloud.models.requests import ConvertDocumentRequest
    except ImportError as e:
        raise OfficeConversionError(
            "Aspose SDK is not installed. Run: pip install aspose-words-cloud"
        ) from e

    words_api = asposewordscloud.WordsApi(client_id, client_secret)
    try:
        with open(input_path, "rb") as doc:
            request = ConvertDocumentRequest(document=doc, format="pdf")
            result = words_api.convert_document(request)
    except Exception as e:
        raise OfficeConversionError(
            f"Aspose conversion failed for {os.path.basename(input_path)}: {e}"
        ) from e

    with open(expected, "wb") as out:
        out.write(result)
    return expected


def _office_to_pdf_cloudmersive(
    input_path: str, expected: str, api_key: str,
) -> str:
    url = "https://api.cloudmersive.com/convert/autodetect/to/pdf"
    try:
        with open(input_path, "rb") as doc:
            response = requests.post(
                url,
                headers={"Apikey": api_key},
                files={"inputFile": (os.path.basename(input_path), doc)},
                timeout=300,
            )
    except requests.RequestException as e:
        raise OfficeConversionError(
            f"Cloudmersive conversion failed for "
            f"{os.path.basename(input_path)}: {e}"
        ) from e

    if response.status_code != 200:
        detail = (response.text or "").strip()[:500]
        raise OfficeConversionError(
            f"Cloudmersive conversion failed for "
            f"{os.path.basename(input_path)} (HTTP {response.status_code})"
            + (f": {detail}" if detail else ".")
        )

    with open(expected, "wb") as out:
        out.write(response.content)
    return expected
