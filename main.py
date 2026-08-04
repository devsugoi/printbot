#!/usr/bin/env python3
"""Entry point for the Gmail -> Discord -> Printer bot.

Usage:
    python3 main.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import os

from src.config import load_config
from src.discord_bot import PrintBot
from src.state import StateStore


def parse_args():
    parser = argparse.ArgumentParser(description="Gmail print-request bot")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml"
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)

    os.makedirs(config.storage.jobs_dir, exist_ok=True)
    state = StateStore(
        config.storage.state_file,
        config.storage.processed_email_retention_days,
    )

    bot = PrintBot(config, state)
    bot.run(config.discord.bot_token)


if __name__ == "__main__":
    main()
