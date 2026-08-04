"""Shared confirm / cancel / print orchestration.

Both the Discord button-and-modal callbacks and the email-reply watcher
call into this module rather than printing directly, so there's exactly
one place that decides what "approve" or "cancel" means and handles
notifying both channels.

A per-job asyncio.Lock makes sure that if both channels approve (or
reprint) the same job at nearly the same moment, only one of them actually
triggers a print -- the other sees the job has already moved past
"awaiting_confirmation" and is treated as a no-op.

This module doesn't know anything about discord.py directly. Discord
notifications go through the `on_notify_discord` callback, which
discord_bot.py fills in -- that keeps view/embed building (a Discord
concern) out of this file.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

from . import gmail_client, pdf_utils, printer
from .config import AppConfig
from .state import (
    STATUS_AWAITING_CONFIRMATION,
    STATUS_CANCELLED,
    STATUS_CONFIRMED,
    STATUS_FAILED,
    STATUS_PRINTED,
    STATUS_PRINTING,
    PrintJob,
    StateStore,
)

logger = logging.getLogger(__name__)

DiscordNotifier = Callable[[PrintJob, str, Optional[list[str]]], Awaitable[None]]


class ConfirmationManager:
    def __init__(self, app_config: AppConfig, state: StateStore, gmail_service):
        self.app_config = app_config
        self.state = state
        self.gmail_service = gmail_service
        self.on_notify_discord: Optional[DiscordNotifier] = None
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, message_id: str) -> asyncio.Lock:
        if message_id not in self._locks:
            self._locks[message_id] = asyncio.Lock()
        return self._locks[message_id]

    # -- public entry points ------------------------------------------------

    async def handle_approval(
        self,
        message_id: str,
        source: str,
        actor: str,
        copies: Optional[int] = None,
        is_explicit_retry: bool = False,
    ):
        """source is "discord" or "email"; actor is a user tag or email
        address, shown in the "approved via" notification.

        is_explicit_retry distinguishes a dedicated "Print again"/"Reprint"
        action from a plain confirmation of the currently pending ask. This
        matters for the race where both channels respond to the SAME ask
        at nearly the same moment: whichever one the lock lets through
        first prints the job and moves it to "printed"; the other one,
        once it finally gets the lock, would otherwise see a
        reprintable status and incorrectly print again. Requiring an
        explicit retry flag for anything other than "still awaiting
        confirmation" closes that gap -- a second, non-explicit approval
        for a job that's no longer awaiting confirmation is treated as
        redundant rather than a new request.
        """
        async with self._lock_for(message_id):
            job = self.state.get_job(message_id)
            if job is None:
                return

            currently_awaiting = job.status == STATUS_AWAITING_CONFIRMATION
            retry_eligible = is_explicit_retry and job.status in (STATUS_PRINTED, STATUS_FAILED)

            if not (currently_awaiting or retry_eligible):
                logger.info(
                    "Ignoring redundant approval for job %s (status=%s, via=%s/%s)",
                    message_id, job.status, source, actor,
                )
                return

            if job.status == STATUS_PRINTED:
                # A full "print again" after a previous success starts over.
                job.current_group_index = 0

            job.copies = copies if copies is not None else (job.copies or 1)
            job.status = STATUS_CONFIRMED
            job.confirmed_via = source
            job.confirmed_by = actor
            self.state.save_job(job)

            plural = "y" if job.copies == 1 else "ies"
            await self.notify_both(
                job,
                discord_text=(
                    f"✅ Approved via {source} ({actor}) — printing "
                    f"{job.copies} cop{plural}."
                ),
                email_text=(
                    f"Approved via {source} ({actor}). Printing "
                    f"{job.copies} cop{plural} now."
                ),
            )

            await self._print_job(job)

    async def handle_cancel(self, message_id: str, source: str, actor: str):
        async with self._lock_for(message_id):
            job = self.state.get_job(message_id)
            if job is None or job.status != STATUS_AWAITING_CONFIRMATION:
                logger.info(
                    "Ignoring cancel for job %s (not awaiting confirmation)",
                    message_id,
                )
                return

            job.status = STATUS_CANCELLED
            job.confirmed_via = source
            job.confirmed_by = actor
            self.state.save_job(job)

            await self.notify_both(
                job,
                discord_text=f"🚫 Cancelled via {source} ({actor}).",
                email_text=(
                    f"Cancelled via {source} ({actor}). This job will not "
                    f"be printed."
                ),
            )

    # -- printing -------------------------------------------------------

    async def _print_job(self, job: PrintJob):
        job.status = STATUS_PRINTING
        self.state.save_job(job)

        available = await asyncio.to_thread(
            printer.is_printer_available, self.app_config.printer.name
        )
        if not available:
            await self._fail_job(
                job, "Printer not detected. Is it powered on and connected?"
            )
            return

        groups = job.paper_size_groups()

        for group_index in range(job.current_group_index, len(groups)):
            paper_size, files = groups[group_index]

            job.attempts += 1
            job.last_attempt_at = time.time()
            self.state.save_job(job)

            for f in files:
                result = await asyncio.to_thread(
                    printer.print_file,
                    f.path, paper_size, self.app_config.printer.name, job.copies,
                )
                if not result.success:
                    await self._fail_job(job, result.message, group_index)
                    return

            job.current_group_index = group_index + 1
            self.state.save_job(job)

            if job.current_group_index < len(groups):
                # More paper-size groups remain -- pause for a tray swap
                # and re-enter the confirmation flow for the next group.
                next_paper_size, _ = groups[job.current_group_index]
                job.status = STATUS_AWAITING_CONFIRMATION
                self.state.save_job(job)
                await self.notify_both(
                    job,
                    discord_text=(
                        f"📄 Printed the {pdf_utils.PAPER_SIZE_LABELS[paper_size]} "
                        f"part of **{job.subject}**. The rest needs "
                        f"{pdf_utils.PAPER_SIZE_LABELS[next_paper_size]} — swap "
                        f"the tray, then confirm to continue."
                    ),
                    email_text=(
                        f"Printed the {pdf_utils.PAPER_SIZE_LABELS[paper_size]} "
                        f"part. The rest of this job needs "
                        f"{pdf_utils.PAPER_SIZE_LABELS[next_paper_size]} — please "
                        f"swap the paper in the tray, then reply to confirm."
                    ),
                )
                return

        job.status = STATUS_PRINTED
        self.state.save_job(job)
        plural = "y" if job.copies == 1 else "ies"
        await self.notify_both(
            job,
            discord_text=(
                f"✅ Printed **{job.subject}** successfully "
                f"({job.copies} cop{plural})."
            ),
            email_text=(
                f"Printed successfully ({job.copies} cop{plural}). Reply "
                f'"print again" any time to print another copy.'
            ),
        )

    async def _fail_job(self, job: PrintJob, error_message: str, group_index: Optional[int] = None):
        if group_index is not None:
            # Resume from this same group next time rather than skipping
            # ahead or restarting groups that already succeeded.
            job.current_group_index = group_index
        job.status = STATUS_FAILED
        job.last_error = error_message
        job.last_attempt_at = time.time()
        self.state.save_job(job)

        await self.notify_both(
            job,
            discord_text=(
                f"❌ Printing **{job.subject}** failed: {error_message}\n"
                f"Click Reprint below, or reply to this email, to try again."
            ),
            email_text=(
                f"Printing failed: {error_message}\n"
                f'Reply "print again" (or use the Reprint button on Discord) '
                f"to try again."
            ),
        )

    # -- notifications ----------------------------------------------------

    async def notify_both(
        self,
        job: PrintJob,
        discord_text: str,
        email_text: str,
        file_paths: Optional[list[str]] = None,
    ):
        """Sends the given text to both channels. One channel failing
        (e.g. Discord being briefly unreachable) doesn't prevent the other
        from being notified."""
        tasks = [self._notify_email(job, email_text, file_paths)]
        if self.on_notify_discord:
            tasks.append(self.on_notify_discord(job, discord_text, file_paths))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _notify_email(self, job: PrintJob, text: str, file_paths: Optional[list[str]]):
        try:
            await asyncio.to_thread(
                gmail_client.send_reply,
                self.gmail_service,
                job.reply_to_address,
                job.subject,
                job.thread_id,
                job.original_rfc_message_id,
                text,
                file_paths,
            )
            job.last_seen_internal_date_ms = int(time.time() * 1000)
            self.state.save_job(job)
        except Exception:
            logger.exception(
                "Failed to send email notification for job %s", job.message_id
            )
