"""Analyze print files for page count and approximate ink coverage."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import fitz
from PIL import Image
from pypdf import PdfReader

from . import pdf_utils
from .config import CostEstimationConfig
from .state import PrintFile

logger = logging.getLogger(__name__)


@dataclass
class PageCoverage:
    bw_coverage: float
    color_coverage: float


@dataclass
class FileAnalysis:
    path: str
    page_count: int
    paper_size: str
    paper_type: str
    pages: list[PageCoverage] = field(default_factory=list)

    @property
    def avg_bw_coverage(self) -> float:
        if not self.pages:
            return 0.0
        return sum(p.bw_coverage for p in self.pages) / len(self.pages)

    @property
    def avg_color_coverage(self) -> float:
        if not self.pages:
            return 0.0
        return sum(p.color_coverage for p in self.pages) / len(self.pages)


def _sample_page_indices(page_count: int, max_pages: int) -> list[int]:
    if page_count <= 0:
        return []
    if page_count <= max_pages:
        return list(range(page_count))
    if max_pages <= 1:
        return [0]
    step = (page_count - 1) / (max_pages - 1)
    indices = sorted({int(round(i * step)) for i in range(max_pages)})
    return indices


def _pixel_chroma(r: int, g: int, b: int) -> int:
    return max(r, g, b) - min(r, g, b)


def _coverage_from_image(
    img: Image.Image,
    *,
    white_threshold: int,
    color_chroma_threshold: int,
    sample_stride: int,
) -> PageCoverage:
    rgb = img.convert("RGB")
    width, height = rgb.size
    stride = max(1, sample_stride)
    sampled = 0
    bw_pixels = 0
    color_pixels = 0

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            r, g, b = rgb.getpixel((x, y))
            sampled += 1
            if r >= white_threshold and g >= white_threshold and b >= white_threshold:
                continue
            if _pixel_chroma(r, g, b) >= color_chroma_threshold:
                color_pixels += 1
            else:
                bw_pixels += 1

    if sampled == 0:
        return PageCoverage(0.0, 0.0)
    return PageCoverage(
        bw_coverage=bw_pixels / sampled,
        color_coverage=color_pixels / sampled,
    )


def _analyze_image_file(path: str, config: CostEstimationConfig) -> list[PageCoverage]:
    with Image.open(path) as img:
        return [
            _coverage_from_image(
                img,
                white_threshold=config.analysis.white_rgb_threshold,
                color_chroma_threshold=config.analysis.color_chroma_threshold,
                sample_stride=config.analysis.pixel_sample_stride,
            )
        ]


def _analyze_pdf_file(
    path: str, config: CostEstimationConfig
) -> tuple[int, list[PageCoverage]]:
    reader = PdfReader(path)
    page_count = len(reader.pages)
    if page_count == 0:
        return 0, []

    sample_indices = _sample_page_indices(
        page_count, config.analysis.max_pages_to_analyze
    )
    scale = config.analysis.render_dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    coverages: list[PageCoverage] = []

    with fitz.open(path) as doc:
        for page_index in sample_indices:
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            coverages.append(
                _coverage_from_image(
                    img,
                    white_threshold=config.analysis.white_rgb_threshold,
                    color_chroma_threshold=config.analysis.color_chroma_threshold,
                    sample_stride=config.analysis.pixel_sample_stride,
                )
            )

    if len(coverages) < page_count:
        avg_bw = sum(c.bw_coverage for c in coverages) / len(coverages)
        avg_color = sum(c.color_coverage for c in coverages) / len(coverages)
        coverages = [
            PageCoverage(bw_coverage=avg_bw, color_coverage=avg_color)
            for _ in range(page_count)
        ]

    return page_count, coverages


def _infer_paper_type(
    pages: list[PageCoverage],
    config: CostEstimationConfig,
) -> str:
    if not pages:
        return config.default_paper_type

    avg_color = sum(p.color_coverage for p in pages) / len(pages)
    if avg_color >= config.photo_color_coverage_threshold:
        return "photo"

    pages_above = sum(
        1 for p in pages if p.color_coverage >= config.photo_color_coverage_threshold
    )
    if pages_above > len(pages) / 2:
        return "photo"
    return config.default_paper_type


def analyze_print_file(
    print_file: PrintFile,
    config: CostEstimationConfig,
) -> FileAnalysis:
    path = print_file.path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Print file not found: {path}")

    if pdf_utils.is_pdf_file(path):
        page_count, pages = _analyze_pdf_file(path, config)
        if page_count == 0:
            raise ValueError(f"PDF has no pages: {path}")
        paper_type = _infer_paper_type(pages, config)
        return FileAnalysis(
            path=path,
            page_count=page_count,
            paper_size=print_file.paper_size,
            paper_type=paper_type,
            pages=pages,
        )

    if pdf_utils.is_image_file(path):
        pages = _analyze_image_file(path, config)
        paper_type = _infer_paper_type(pages, config)
        return FileAnalysis(
            path=path,
            page_count=1,
            paper_size=print_file.paper_size,
            paper_type=paper_type,
            pages=pages,
        )

    raise ValueError(f"Unsupported file type for cost analysis: {path}")


def analyze_print_files(
    files: list[PrintFile],
    config: CostEstimationConfig,
) -> tuple[list[FileAnalysis], list[str]]:
    analyses: list[FileAnalysis] = []
    warnings: list[str] = []

    for print_file in files:
        try:
            analyses.append(analyze_print_file(print_file, config))
        except Exception as exc:
            name = os.path.basename(print_file.path)
            message = f"Could not analyze {name}: {exc}"
            logger.warning(message)
            warnings.append(message)

    return analyses, warnings
