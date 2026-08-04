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
class AppConfig:
    gmail: GmailConfig
    gemini: GeminiConfig
    discord: DiscordConfig
    printer: PrinterConfig
    storage: StorageConfig


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
    )
