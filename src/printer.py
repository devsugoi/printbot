"""Printing via CUPS (the standard Linux printing system, used to drive the
Brother DCP-J100 over USB or Wi-Fi on the Pi).

Requires CUPS + the Brother driver to already be installed and the printer
already added (see README) -- this module just shells out to `lp`.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Maps our internal paper size names to CUPS "media" option values.
# "Short" = short bond paper = Letter (8.5x11in), "Long" = long bond paper
# = Legal (8.5x14in). These are the common CUPS/IPP standard names; run
# `lpoptions -p <printer> -l` on your Pi to confirm what your driver
# actually supports and adjust if needed (some drivers expect e.g.
# "na_letter_8.5x11in" / "na_legal_8.5x14in" instead).
CUPS_MEDIA_NAMES = {
    "Short": "Letter",
    "Long": "Legal",
}


@dataclass
class PrintResult:
    success: bool
    message: str


def is_printer_available(printer_name: str) -> bool:
    try:
        result = subprocess.run(
            ["lpstat", "-p", printer_name],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error("Could not query printer status: %s", e)
        return False


def print_file(filepath: str, paper_size: str, printer_name: str, copies: int = 1) -> PrintResult:
    media = CUPS_MEDIA_NAMES.get(paper_size, paper_size)
    copies = max(1, int(copies))

    command = [
        "lp",
        "-d", printer_name,
        "-o", f"media={media}",
        "-o", "fit-to-page",
        "-n", str(copies),
        filepath,
    ]

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return PrintResult(False, "Print command timed out.")
    except FileNotFoundError:
        return PrintResult(
            False, "`lp` command not found -- is CUPS installed on this Pi?"
        )

    if result.returncode == 0:
        return PrintResult(True, result.stdout.strip() or "Sent to printer.")

    error_message = result.stderr.strip() or result.stdout.strip() or "Unknown error"
    return PrintResult(False, error_message)
