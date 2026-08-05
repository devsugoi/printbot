"""Simple JSON-file-backed persistence.

Tracks which Gmail messages have already been looked at, and keeps a record
of every print job -- files (each with its own paper size), confirmation
state, email-thread info, and copy count -- so a job can be re-printed or
re-confirmed later, even after the bot restarts and the original Discord
buttons have expired.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

# Job status values:
#   awaiting_confirmation -> confirmed -> printing -> printed
#                                                   -> failed (resumable)
#                         -> cancelled (terminal)
# "printed" and "failed" both remain eligible for a later "print again" /
# "reprint" approval, which re-enters the same confirm/print path.
STATUS_AWAITING_CONFIRMATION = "awaiting_confirmation"
STATUS_CONFIRMED = "confirmed"
STATUS_PRINTING = "printing"
STATUS_PRINTED = "printed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

# Statuses for which a new approval (Discord click or email reply) is
# meaningful -- i.e. re-entering the confirm/print flow instead of being
# ignored as "already handled".
REPRINTABLE_STATUSES = (STATUS_AWAITING_CONFIRMATION, STATUS_PRINTED, STATUS_FAILED)


@dataclass
class PrintOptions:
    """Optional print instructions set at approval time (email or Discord)."""
    page_ranges: Optional[str] = None       # CUPS format: "2", "1-3", "1,3-5"
    paper_size_override: Optional[str] = None  # canonical name, e.g. "A4", "Short"


@dataclass
class PrintFile:
    path: str
    paper_size: str            # "Short" or "Long"
    is_generated: bool = False  # True if the bot built this file itself
                                 # (e.g. images combined into one PDF)
    scaled_path: Optional[str] = None  # Letter-fit PDF when Long -> Short
    office_source_path: Optional[str] = None  # original office file if path is a converted PDF
    conversion_backend: Optional[str] = None  # libreoffice | aspose | cloudmersive


@dataclass
class PrintJob:
    message_id: str
    thread_id: str
    subject: str
    sender: str                          # original "From" header, for display
    reply_to_address: str                # address the bot replies to in-thread
    owner_email: str                     # the mailbox owner's own address
    original_rfc_message_id: str         # Message-Id header of the source email
    files: list[PrintFile]
    status: str = STATUS_AWAITING_CONFIRMATION
    copies: int = 1
    current_group_index: int = 0         # which paper-size group prints next
    confirmed_via: Optional[str] = None  # "discord" | "email"
    confirmed_by: Optional[str] = None   # discord user tag or email address
    last_seen_internal_date_ms: int = 0  # for detecting new email replies
    created_at: float = field(default_factory=time.time)
    last_attempt_at: Optional[float] = None
    last_error: Optional[str] = None
    attempts: int = 0
    fit_long_on_short: bool = False  # scale Long content onto Short paper
    approval_options: PrintOptions = field(default_factory=PrintOptions)

    def paper_size_groups(self) -> list[tuple[str, list[PrintFile]]]:
        """Groups files by paper size, preserving first-seen order. Most
        jobs have exactly one group; a job only has more than one when a
        single email mixed images (always Short) with a document that
        needed Long."""
        groups: list[tuple[str, list[PrintFile]]] = []
        seen = {}
        for f in self.files:
            if f.paper_size not in seen:
                seen[f.paper_size] = len(groups)
                groups.append((f.paper_size, []))
            groups[seen[f.paper_size]][1].append(f)
        return groups

    def generated_files(self) -> list[PrintFile]:
        return [f for f in self.files if f.is_generated]

    def has_long_paper_pending(self) -> bool:
        """True if any not-yet-printed group needs long bond paper."""
        groups = self.paper_size_groups()
        if self.current_group_index >= len(groups):
            return False
        return any(g[0] == "Long" for g in groups[self.current_group_index:])


_PRINT_JOB_FIELDS = {f.name for f in fields(PrintJob)}


def _job_to_dict(job: PrintJob) -> dict:
    return asdict(job)


def _job_from_dict(raw: dict) -> PrintJob:
    raw = dict(raw)
    raw["files"] = [PrintFile(**f) for f in raw.get("files", [])]
    opts = raw.get("approval_options") or {}
    if isinstance(opts, dict):
        raw["approval_options"] = PrintOptions(**opts)
    filtered = {k: v for k, v in raw.items() if k in _PRINT_JOB_FIELDS}
    return PrintJob(**filtered)


class StateStore:
    """Thread-safe wrapper around a JSON file.

    The Gmail polling loop, the email-reply watcher, and Discord button/
    modal callbacks can all touch this at once, so every read-modify-write
    is guarded by a lock.
    """

    def __init__(self, state_file: str, processed_retention_days: int = 30):
        self.state_file = state_file
        self._processed_retention_days = processed_retention_days
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.state_file):
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"processed_message_ids": {}, "jobs": {}}

        processed = data.get("processed_message_ids", {})
        if isinstance(processed, list):
            now = time.time()
            data["processed_message_ids"] = {mid: now for mid in processed}
        elif not isinstance(processed, dict):
            data["processed_message_ids"] = {}
        if "jobs" not in data:
            data["jobs"] = {}
        return data

    def _save(self):
        tmp_path = self.state_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp_path, self.state_file)

    def _prune_processed_locked(self):
        if self._processed_retention_days <= 0:
            return
        cutoff = time.time() - self._processed_retention_days * 86400
        processed = self._data["processed_message_ids"]
        self._data["processed_message_ids"] = {
            mid: ts for mid, ts in processed.items() if ts >= cutoff
        }

    # -- processed messages -------------------------------------------------

    def is_processed(self, message_id: str) -> bool:
        with self._lock:
            return message_id in self._data["processed_message_ids"]

    def mark_processed(self, message_id: str):
        with self._lock:
            self._data["processed_message_ids"][message_id] = time.time()
            self._prune_processed_locked()
            self._save()

    # -- print jobs -----------------------------------------------------

    def save_job(self, job: PrintJob):
        with self._lock:
            self._data["jobs"][job.message_id] = _job_to_dict(job)
            self._save()

    def get_job(self, message_id: str) -> Optional[PrintJob]:
        with self._lock:
            raw = self._data["jobs"].get(message_id)
            return _job_from_dict(raw) if raw else None

    def update_last_seen_internal_date(self, message_id: str, internal_date_ms: int):
        """Update only the reply watermark for a job, without overwriting
        other fields that may have changed concurrently."""
        with self._lock:
            raw = self._data["jobs"].get(message_id)
            if raw is None:
                return
            raw["last_seen_internal_date_ms"] = max(
                raw.get("last_seen_internal_date_ms", 0), internal_date_ms
            )
            self._save()

    def all_jobs(self) -> list[PrintJob]:
        with self._lock:
            return [_job_from_dict(raw) for raw in self._data["jobs"].values()]

    def find_job_by_thread(self, thread_id: str) -> Optional[PrintJob]:
        """Returns the earliest-created job for a Gmail thread, if any.
        Used to guarantee at most one job per thread -- later messages in
        the thread (replies, the bot's own notifications) must never spawn
        duplicate jobs that would then approve each other."""
        with self._lock:
            matches = [
                _job_from_dict(raw)
                for raw in self._data["jobs"].values()
                if raw.get("thread_id") == thread_id
            ]
        if not matches:
            return None
        return min(matches, key=lambda j: j.created_at)

    def jobs_awaiting_email_replies(self, retention_days: int) -> list[PrintJob]:
        """Jobs whose email thread is still worth polling for new replies:
        anything reprintable, and not older than the configured retention
        window."""
        cutoff = time.time() - retention_days * 86400
        return [
            job for job in self.all_jobs()
            if job.status in REPRINTABLE_STATUSES and job.created_at >= cutoff
        ]
