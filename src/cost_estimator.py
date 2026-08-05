"""Compute and format PHP print cost estimates from file analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .config import CostEstimationConfig
from .file_analysis import FileAnalysis, analyze_print_files
from .state import PrintFile


@dataclass
class CostLine:
    label: str
    amount: float


@dataclass
class CostEstimate:
    lines: list[CostLine]
    subtotal: float
    markup_multiplier: float
    total: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _paper_price(
    config: CostEstimationConfig,
    paper_type: str,
    paper_size: str,
) -> float:
    type_prices = config.paper_prices.get(paper_type, {})
    if paper_size in type_prices:
        return type_prices[paper_size]
    bond_prices = config.paper_prices.get("bond", {})
    return bond_prices.get(paper_size, 0.0)


def _paper_size_label(paper_size: str) -> str:
    labels = {
        "Short": "Short bond",
        "Long": "Long bond",
        "A4": "A4",
    }
    return labels.get(paper_size, paper_size)


def _paper_type_label(paper_type: str) -> str:
    return paper_type.capitalize()


def _format_money(amount: float, symbol: str) -> str:
    return f"{symbol}{amount:,.2f}"


def _ink_cost_for_analysis(
    analysis: FileAnalysis,
    config: CostEstimationConfig,
) -> float:
    total = 0.0
    for page in analysis.pages:
        total += page.bw_coverage * config.ink.bw_cost_per_full_page
        total += page.color_coverage * config.ink.color_cost_per_full_page
    return total


def estimate_from_analyses(
    analyses: list[FileAnalysis],
    config: CostEstimationConfig,
    warnings: list[str] | None = None,
) -> CostEstimate | None:
    if not analyses:
        return None

    paper_groups: dict[tuple[str, str], int] = {}
    total_ink = 0.0
    total_pages = 0
    weighted_color = 0.0
    weighted_bw = 0.0

    for analysis in analyses:
        key = (analysis.paper_size, analysis.paper_type)
        paper_groups[key] = paper_groups.get(key, 0) + analysis.page_count
        total_ink += _ink_cost_for_analysis(analysis, config)
        for page in analysis.pages:
            total_pages += 1
            weighted_color += page.color_coverage
            weighted_bw += page.bw_coverage

    paper_cost = 0.0
    lines: list[CostLine] = []
    for (paper_size, paper_type), page_count in sorted(paper_groups.items()):
        sheet_price = _paper_price(config, paper_type, paper_size)
        amount = sheet_price * page_count
        paper_cost += amount
        label = (
            f"Paper ({_paper_size_label(paper_size)} {_paper_type_label(paper_type)} "
            f"× {page_count} page{'s' if page_count != 1 else ''})"
        )
        lines.append(CostLine(label=label, amount=amount))

    avg_color_pct = (weighted_color / total_pages * 100) if total_pages else 0.0
    avg_bw_pct = (weighted_bw / total_pages * 100) if total_pages else 0.0
    if avg_color_pct >= avg_bw_pct and avg_color_pct > 0:
        ink_label = f"Ink (Color ~{avg_color_pct:.0f}% avg)"
    elif avg_bw_pct > 0:
        ink_label = f"Ink (B&W ~{avg_bw_pct:.0f}% avg)"
    else:
        ink_label = "Ink"
    lines.append(CostLine(label=ink_label, amount=total_ink))

    subtotal = paper_cost + total_ink
    total = subtotal * config.markup_multiplier
    return CostEstimate(
        lines=lines,
        subtotal=subtotal,
        markup_multiplier=config.markup_multiplier,
        total=total,
        warnings=list(warnings or []),
    )


def estimate_job(
    files: list[PrintFile],
    config: CostEstimationConfig,
) -> CostEstimate | None:
    analyses, warnings = analyze_print_files(files, config)
    return estimate_from_analyses(analyses, config, warnings)


def format_estimate_discord(
    estimate: CostEstimate,
    config: CostEstimationConfig,
) -> str:
    symbol = config.currency_symbol
    lines = [
        "📄 **Print Cost Estimate** (1 copy)",
    ]
    for line in estimate.lines:
        lines.append(f"- {line.label}: {_format_money(line.amount, symbol)}")
    lines.append(
        f"- **Total Estimated Price: {_format_money(estimate.total, symbol)}**"
    )
    if estimate.markup_multiplier != 1.0:
        lines.append(
            f"- Includes markup ×{estimate.markup_multiplier:g}"
        )
    lines.append("_Estimate is for 1 copy; multiply by N for N copies._")
    lines.append("_Estimate only; final price may vary._")
    for warning in estimate.warnings:
        lines.append(f"⚠️ {warning}")
    return "\n".join(lines)


def format_estimate_email(
    estimate: CostEstimate,
    config: CostEstimationConfig,
) -> str:
    symbol = config.currency_symbol
    lines = [
        "Print Cost Estimate (1 copy):",
    ]
    for line in estimate.lines:
        lines.append(f"- {line.label}: {_format_money(line.amount, symbol)}")
    lines.append(
        f"Total estimated price: {_format_money(estimate.total, symbol)}"
    )
    if estimate.markup_multiplier != 1.0:
        lines.append(f"(Includes markup x{estimate.markup_multiplier:g})")
    lines.append(
        "Estimate is for 1 copy; multiply by N for N copies. "
        "Final price may vary."
    )
    for warning in estimate.warnings:
        lines.append(f"Note: {warning}")
    return "\n".join(lines)
