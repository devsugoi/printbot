"""Loads and validates config.yaml into simple, typed objects.

Any string value in the YAML file can be written as "ENV:VAR_NAME" to be
resolved from an environment variable instead of being stored in plain text.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml


def _resolve(value):
    """Resolve "ENV:VAR_NAME" strings to environment variables. Leaves
    everything else untouched (including lists, which are resolved
    element-wise)."""
    if isinstance(value, str) and value.startswith("ENV:"):
        var_name = value[len("ENV:"):]
        resolved = os.environ.get(var_name)
        if not resolved:
            raise ValueError(
                f"Config referenced environment variable '{var_name}' "
                f"but it is not set."
            )
        return resolved
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value


@dataclass
class GmailConfig:
    credentials_file: str
    token_file: str
    search_query: str
    poll_interval_seconds: int
    # Security settings for email-based approval (see README):
    # - Your own address can always approve/cancel/reprint.
    # - If approved_reply_senders is non-empty, ONLY those addresses (plus
    #   your own) may approve -- this overrides allow_non_owner_email_approval.
    # - Otherwise, allow_non_owner_email_approval controls whether ANY
    #   reply in the thread counts.
    allow_non_owner_email_approval: bool = False
    approved_reply_senders: list[str] = field(default_factory=list)


@dataclass
class GeminiConfig:
    api_keys: list[str]
    models: list[str]


@dataclass
class DiscordConfig:
    bot_token: str
    user_id: int
    channel_id: int
    command_prefix: str = "!"


@dataclass
class PrinterConfig:
    name: str
    default_paper_size: str
    supported_paper_sizes: list[str] = field(default_factory=list)


@dataclass
class StorageConfig:
    jobs_dir: str
    state_file: str
    processed_email_retention_days: int = 30


@dataclass
class AsposeConfig:
    enabled: bool = False
    client_id: str = ""
    client_secret: str = ""

    def is_available(self) -> bool:
        return self.enabled and bool(self.client_id) and bool(self.client_secret)


@dataclass
class CloudmersiveConfig:
    enabled: bool = False
    api_key: str = ""

    def is_available(self) -> bool:
        return self.enabled and bool(self.api_key)


@dataclass
class OfficeConversionConfig:
    aspose: AsposeConfig = field(default_factory=AsposeConfig)
    cloudmersive: CloudmersiveConfig = field(default_factory=CloudmersiveConfig)


_DEFAULT_PAPER_PRICES: dict[str, dict[str, float]] = {
    "bond": {"Short": 1.50, "Long": 2.00, "A4": 1.75},
    "photo": {"Short": 20.00, "Long": 25.00, "A4": 22.00},
}


@dataclass
class CostAnalysisConfig:
    render_dpi: int = 72
    pixel_sample_stride: int = 4
    white_rgb_threshold: int = 245
    color_chroma_threshold: int = 20
    max_pages_to_analyze: int = 20


@dataclass
class CostInkConfig:
    bw_cost_per_full_page: float = 2.00
    color_cost_per_full_page: float = 8.00


@dataclass
class CostEstimationConfig:
    enabled: bool = False
    currency_symbol: str = "₱"
    markup_multiplier: float = 1.0
    default_paper_type: str = "bond"
    photo_color_coverage_threshold: float = 0.25
    paper_prices: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            k: dict(v) for k, v in _DEFAULT_PAPER_PRICES.items()
        }
    )
    ink: CostInkConfig = field(default_factory=CostInkConfig)
    analysis: CostAnalysisConfig = field(default_factory=CostAnalysisConfig)


@dataclass
class AppConfig:
    gmail: GmailConfig
    gemini: GeminiConfig
    discord: DiscordConfig
    printer: PrinterConfig
    storage: StorageConfig
    office_conversion: OfficeConversionConfig = field(
        default_factory=OfficeConversionConfig
    )
    cost_estimation: CostEstimationConfig = field(
        default_factory=CostEstimationConfig
    )


def load_config(path: str = "config.yaml") -> AppConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file '{path}' not found. Copy config.example.yaml to "
            f"{path} and fill in your details."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    gmail = raw["gmail"]
    gemini = raw["gemini"]
    discord = raw["discord"]
    printer = raw["printer"]
    storage = raw["storage"]
    office = raw.get("office_conversion", {})
    aspose = office.get("aspose", {})
    cloudmersive = office.get("cloudmersive", {})
    cost = raw.get("cost_estimation", {})
    ink = cost.get("ink", {})
    analysis = cost.get("analysis", {})
    paper_prices_raw = cost.get("paper_prices")
    if paper_prices_raw:
        paper_prices = {
            str(ptype): {str(size): float(price) for size, price in sizes.items()}
            for ptype, sizes in paper_prices_raw.items()
        }
    else:
        paper_prices = {
            k: dict(v) for k, v in _DEFAULT_PAPER_PRICES.items()
        }

    return AppConfig(
        gmail=GmailConfig(
            credentials_file=_resolve(gmail["credentials_file"]),
            token_file=_resolve(gmail["token_file"]),
            search_query=_resolve(gmail["search_query"]),
            poll_interval_seconds=int(gmail["poll_interval_seconds"]),
            allow_non_owner_email_approval=bool(
                gmail.get("allow_non_owner_email_approval", False)
            ),
            approved_reply_senders=[
                s.lower() for s in _resolve(gmail.get("approved_reply_senders", []))
            ],
        ),
        gemini=GeminiConfig(
            api_keys=_resolve(gemini["api_keys"]),
            models=_resolve(gemini["models"]),
        ),
        discord=DiscordConfig(
            bot_token=_resolve(discord["bot_token"]),
            user_id=int(discord["user_id"]),
            channel_id=int(discord["channel_id"]),
            command_prefix=discord.get("command_prefix", "!"),
        ),
        printer=PrinterConfig(
            name=_resolve(printer["name"]),
            default_paper_size=printer["default_paper_size"],
            supported_paper_sizes=printer.get("supported_paper_sizes", []),
        ),
        storage=StorageConfig(
            jobs_dir=storage["jobs_dir"],
            state_file=storage["state_file"],
            processed_email_retention_days=int(
                storage.get("processed_email_retention_days", 30)
            ),
        ),
        office_conversion=OfficeConversionConfig(
            aspose=AsposeConfig(
                enabled=bool(aspose.get("enabled", False)),
                client_id=(
                    _resolve(aspose["client_id"])
                    if aspose.get("enabled") and aspose.get("client_id")
                    else ""
                ),
                client_secret=(
                    _resolve(aspose["client_secret"])
                    if aspose.get("enabled") and aspose.get("client_secret")
                    else ""
                ),
            ),
            cloudmersive=CloudmersiveConfig(
                enabled=bool(cloudmersive.get("enabled", False)),
                api_key=(
                    _resolve(cloudmersive["api_key"])
                    if cloudmersive.get("enabled") and cloudmersive.get("api_key")
                    else ""
                ),
            ),
        ),
        cost_estimation=CostEstimationConfig(
            enabled=bool(cost.get("enabled", False)),
            currency_symbol=str(cost.get("currency_symbol", "₱")),
            markup_multiplier=float(cost.get("markup_multiplier", 1.0)),
            default_paper_type=str(cost.get("default_paper_type", "bond")),
            photo_color_coverage_threshold=float(
                cost.get("photo_color_coverage_threshold", 0.25)
            ),
            paper_prices=paper_prices,
            ink=CostInkConfig(
                bw_cost_per_full_page=float(
                    ink.get("bw_cost_per_full_page", 2.00)
                ),
                color_cost_per_full_page=float(
                    ink.get("color_cost_per_full_page", 8.00)
                ),
            ),
            analysis=CostAnalysisConfig(
                render_dpi=int(analysis.get("render_dpi", 72)),
                pixel_sample_stride=int(analysis.get("pixel_sample_stride", 4)),
                white_rgb_threshold=int(analysis.get("white_rgb_threshold", 245)),
                color_chroma_threshold=int(analysis.get("color_chroma_threshold", 20)),
                max_pages_to_analyze=int(analysis.get("max_pages_to_analyze", 20)),
            ),
        ),
    )
