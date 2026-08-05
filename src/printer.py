"""Printing via CUPS (the standard Linux printing system, used to drive the
Brother DCP-J100 over USB or Wi-Fi on the Pi).

Requires CUPS + the Brother driver to already be installed and the printer
already added (see README) -- this module just shells out to `lp`.

Note: a successful `lp` only means CUPS *accepted* the job into its queue.
It does not guarantee the physical printer has finished (or even started)
printing. Check journal logs for the CUPS job id, then use `lpstat -o` /
`lpstat -W completed` on the Pi if paper never comes out.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
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
    "A4": "A4",
}

# Substrings in `lpstat -p` output that mean the queue exists but is not
# ready to print. `lpstat` often still exits 0 in these cases.
_UNAVAILABLE_MARKERS = (
    "disabled",
    "unable to",
    "not ready",
    "offline",
    "paused",
)

# `lp` success output looks like: 'request id is DCPJ100-42 (1 file(s))'
_LP_REQUEST_ID_RE = re.compile(r"request id is (\S+-\d+)")

# How long to wait for a submitted job to leave the CUPS queue before
# concluding it is stuck, and how often to check. The DCP-J100 is a slow
# inkjet; multi-page / image-heavy PDFs routinely take well over a minute.
JOB_COMPLETION_TIMEOUT_SECONDS = 300
JOB_POLL_INTERVAL_SECONDS = 2


@dataclass
class PrintResult:
    success: bool
    message: str


@dataclass
class PrinterAvailability:
    available: bool
    detail: str


def check_printer_availability(printer_name: str) -> PrinterAvailability:
    """Query CUPS for the named printer and decide whether it looks ready."""
    try:
        result = subprocess.run(
            ["lpstat", "-p", printer_name],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        detail = "`lpstat` not found -- is CUPS installed on this Pi?"
        logger.error("Printer check failed for %s: %s", printer_name, detail)
        return PrinterAvailability(False, detail)
    except subprocess.SubprocessError as e:
        detail = f"Could not query printer status: {e}"
        logger.error("Printer check failed for %s: %s", printer_name, detail)
        return PrinterAvailability(False, detail)

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    combined = f"{stdout}\n{stderr}".strip()
    logger.info(
        "Printer check for %s: returncode=%s stdout=%r stderr=%r",
        printer_name, result.returncode, stdout, stderr,
    )

    if result.returncode != 0:
        detail = combined or f"lpstat exited with code {result.returncode}"
        logger.warning("Printer %s not detected: %s", printer_name, detail)
        return PrinterAvailability(False, detail)

    lowered = combined.lower()
    for marker in _UNAVAILABLE_MARKERS:
        if marker in lowered:
            detail = combined or f"Printer reported as {marker}"
            logger.warning(
                "Printer %s found but not ready (%s): %s",
                printer_name, marker, detail,
            )
            return PrinterAvailability(False, detail)

    logger.info("Printer %s detected and looks ready: %s", printer_name, stdout or "(no lpstat text)")
    return PrinterAvailability(True, stdout or "Printer is available.")


def is_printer_available(printer_name: str) -> bool:
    return check_printer_availability(printer_name).available


def _printer_state_message(printer_name: str) -> str:
    """Returns the long-form `lpstat -p <printer> -l` output, which includes
    the printer-state-message CUPS records when a job gets stuck."""
    try:
        result = subprocess.run(
            ["lpstat", "-p", printer_name, "-l"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        return f"(could not read printer state: {e})"
    return (result.stdout or "").strip() or (result.stderr or "").strip()


def wait_for_job(
    printer_name: str,
    job_id: str,
    timeout: float = JOB_COMPLETION_TIMEOUT_SECONDS,
) -> PrintResult:
    """Waits for a submitted CUPS job to leave the not-completed queue.

    `lp` exiting 0 only means the job was queued; a broken driver/filter
    or a held queue can leave it stuck (or silently eat it) while the bot
    would otherwise report success. This polls `lpstat -W not-completed`
    until the job disappears from the queue or the timeout elapses.
    """
    deadline = time.monotonic() + timeout

    while True:
        try:
            result = subprocess.run(
                ["lpstat", "-W", "not-completed", "-o", printer_name],
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error("Could not check status of job %s: %s", job_id, e)
            # Can't verify -- report the submission as-is rather than
            # failing a job that may well be printing.
            return PrintResult(
                True, f"Job {job_id} was queued (status check unavailable: {e})."
            )

        pending = result.stdout or ""
        if job_id not in pending:
            logger.info("CUPS reports job %s completed.", job_id)
            return PrintResult(True, f"CUPS reports job {job_id} completed.")

        if time.monotonic() >= deadline:
            state = _printer_state_message(printer_name)
            logger.error(
                "Job %s still in the CUPS queue after %.0fs. Printer state: %s",
                job_id, timeout, state,
            )
            return PrintResult(
                False,
                f"Job {job_id} was queued but hasn't printed after "
                f"{timeout:.0f}s. Printer state: {state}",
            )

        time.sleep(JOB_POLL_INTERVAL_SECONDS)


def print_file(
    filepath: str,
    paper_size: str,
    printer_name: str,
    copies: int = 1,
    page_ranges: str | None = None,
) -> PrintResult:
    media = CUPS_MEDIA_NAMES.get(paper_size, paper_size)
    copies = max(1, int(copies))
    abs_path = os.path.abspath(filepath)

    logger.info(
        "Preparing print: printer=%s paper_size=%s media=%s copies=%d "
        "page_ranges=%s file=%s",
        printer_name, paper_size, media, copies, page_ranges, abs_path,
    )

    if not os.path.exists(abs_path):
        message = f"Print file does not exist: {abs_path}"
        logger.error(message)
        return PrintResult(False, message)
    if os.path.getsize(abs_path) == 0:
        message = f"Print file is empty: {abs_path}"
        logger.error(message)
        return PrintResult(False, message)

    command = [
        "lp",
        "-d", printer_name,
        "-o", f"media={media}",
        "-o", "fit-to-page",
        "-n", str(copies),
    ]
    if page_ranges:
        command.extend(["-o", f"page-ranges={page_ranges}"])
    command.append(abs_path)
    logger.info("Running print command: %s", command)

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        logger.error("Print command timed out for %s", abs_path)
        return PrintResult(False, "Print command timed out.")
    except FileNotFoundError:
        message = "`lp` command not found -- is CUPS installed on this Pi?"
        logger.error(message)
        return PrintResult(False, message)

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    logger.info(
        "lp finished: returncode=%s stdout=%r stderr=%r",
        result.returncode, stdout, stderr,
    )

    if result.returncode == 0:
        logger.info(
            "CUPS accepted print job for %s (%s): %s",
            abs_path, printer_name, stdout or "Sent to printer.",
        )
        match = _LP_REQUEST_ID_RE.search(stdout)
        if match is None:
            logger.warning(
                "Could not parse a CUPS job id from lp output %r; "
                "skipping completion check.", stdout,
            )
            return PrintResult(True, stdout or "Sent to printer.")
        return wait_for_job(printer_name, match.group(1))

    error_message = stderr or stdout or "Unknown error"
    logger.error(
        "CUPS rejected print job for %s (%s): %s",
        abs_path, printer_name, error_message,
    )
    return PrintResult(False, error_message)
