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
import os
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
    PrintFile,
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
        fit_long_on_short: Optional[bool] = None,
    ) -> bool:
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
                return False

            currently_awaiting = job.status == STATUS_AWAITING_CONFIRMATION
            retry_eligible = is_explicit_retry and job.status in (STATUS_PRINTED, STATUS_FAILED)

            if not (currently_awaiting or retry_eligible):
                logger.info(
                    "Ignoring redundant approval for job %s (status=%s, via=%s/%s)",
                    message_id, job.status, source, actor,
                )
                return False

            if job.status == STATUS_PRINTED:
                # A full "print again" after a previous success starts over.
                job.current_group_index = 0

            job.copies = copies if copies is not None else (job.copies or 1)
            if fit_long_on_short is not None:
                job.fit_long_on_short = fit_long_on_short
            elif currently_awaiting:
                job.fit_long_on_short = False
            job.status = STATUS_CONFIRMED
            job.confirmed_via = source
            job.confirmed_by = actor
            self.state.save_job(job)

            plural = "y" if job.copies == 1 else "ies"
            mode_note = (
                " on short bond paper (scaled to fit)"
                if job.fit_long_on_short
                else ""
            )
            await self.notify_both(
                job,
                discord_text=(
                    f"✅ Approved via {source} ({actor}) — printing "
                    f"{job.copies} cop{plural}{mode_note}."
                ),
                email_text=(
                    f"Approved via {source} ({actor}). Printing "
                    f"{job.copies} cop{plural}{mode_note} now."
                ),
            )

            await self._print_job(job)
            return True

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

        logger.info(
            "Starting print for job %s subject=%r copies=%d fit_long_on_short=%s "
            "files=%d group_index=%d",
            job.message_id, job.subject, job.copies, job.fit_long_on_short,
            len(job.files), job.current_group_index,
        )

        availability = await asyncio.to_thread(
            printer.check_printer_availability, self.app_config.printer.name
        )
        if not availability.available:
            logger.warning(
                "Printer unavailable for job %s: %s",
                job.message_id, availability.detail,
            )
            await self._fail_job(
                job,
                f"Printer not ready: {availability.detail}. "
                f"Is it powered on, connected, and enabled in CUPS?",
            )
            return

        logger.info(
            "Printer available for job %s: %s",
            job.message_id, availability.detail,
        )

        groups = job.paper_size_groups()
        if not job.files or not groups:
            logger.error(
                "Job %s has nothing to print (files=%d groups=%d)",
                job.message_id, len(job.files), len(groups),
            )
            await self._fail_job(job, "Nothing to print — no files were prepared for this job.")
            return
        if job.current_group_index >= len(groups):
            logger.error(
                "Job %s has no remaining paper-size groups "
                "(group_index=%d groups=%d)",
                job.message_id, job.current_group_index, len(groups),
            )
            await self._fail_job(
                job,
                "Nothing left to print — all paper-size groups were already completed.",
            )
            return

        for group_index in range(job.current_group_index, len(groups)):
            paper_size, files = groups[group_index]
            logger.info(
                "Job %s printing group %d/%d paper_size=%s (%d file(s))",
                job.message_id, group_index + 1, len(groups), paper_size, len(files),
            )

            job.attempts += 1
            job.last_attempt_at = time.time()
            self.state.save_job(job)

            for f in files:
                try:
                    print_path, effective_size = await self._resolve_print_target(
                        job, f
                    )
                except pdf_utils.OfficeConversionError as e:
                    logger.error(
                        "Office conversion failed for job %s file %s: %s",
                        job.message_id, f.path, e,
                    )
                    await self._fail_job(job, str(e), group_index)
                    return

                logger.info(
                    "Job %s submitting file %s (requested=%s effective=%s)",
                    job.message_id, print_path, f.paper_size, effective_size,
                )
                result = await asyncio.to_thread(
                    printer.print_file,
                    print_path, effective_size,
                    self.app_config.printer.name, job.copies,
                )
                if not result.success:
                    logger.error(
                        "Job %s print failed for %s: %s",
                        job.message_id, print_path, result.message,
                    )
                    await self._fail_job(job, result.message, group_index)
                    return
                logger.info(
                    "Job %s CUPS accepted %s: %s",
                    job.message_id, print_path, result.message,
                )

            job.current_group_index = group_index + 1
            self.state.save_job(job)

            if (
                job.current_group_index < len(groups)
                and not job.fit_long_on_short
            ):
                # More paper-size groups remain -- pause for a tray swap
                # and re-enter the confirmation flow for the next group.
                next_paper_size, _ = groups[job.current_group_index]
                job.status = STATUS_AWAITING_CONFIRMATION
                self.state.save_job(job)
                logger.info(
                    "Job %s pausing for tray swap: next paper_size=%s",
                    job.message_id, next_paper_size,
                )
                short_option = (
                    ' Or click **Print on short bond** / reply "use short '
                    'bond" to scale the rest onto the paper already loaded.'
                )
                await self.notify_both(
                    job,
                    discord_text=(
                        f"📄 Printed the {pdf_utils.PAPER_SIZE_LABELS[paper_size]} "
                        f"part of **{job.subject}**. The rest needs "
                        f"{pdf_utils.PAPER_SIZE_LABELS[next_paper_size]} — swap "
                        f"the tray, then confirm to continue.{short_option}"
                    ),
                    email_text=(
                        f"Printed the {pdf_utils.PAPER_SIZE_LABELS[paper_size]} "
                        f"part. The rest of this job needs "
                        f"{pdf_utils.PAPER_SIZE_LABELS[next_paper_size]} — please "
                        f"swap the paper in the tray, then reply to confirm."
                        f'{short_option.replace("**", "")}'
                    ),
                )
                return

        job.status = STATUS_PRINTED
        self.state.save_job(job)
        plural = "y" if job.copies == 1 else "ies"
        logger.info(
            "Job %s marked printed (CUPS accepted all files; copies=%d). "
            "Physical output is not verified — check the printer / "
            "`lpstat -o` if no paper came out.",
            job.message_id, job.copies,
        )
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

    async def _resolve_print_target(
        self, job: PrintJob, f: PrintFile
    ) -> tuple[str, str]:
        """Returns (filepath, CUPS paper size) for a file, converting office
        documents to PDF and scaling Long PDFs onto Short when the user
        chose fit_long_on_short.

        Raises pdf_utils.OfficeConversionError if an office file can't be
        converted (e.g. LibreOffice missing)."""
        path = f.path
        if pdf_utils.is_office_file(path):
            # Jobs prepared before office->PDF conversion existed (or whose
            # conversion failed at prepare time) still point at the raw
            # office file -- convert here so reprints of those jobs work.
            converted = await asyncio.to_thread(
                pdf_utils.office_to_pdf, path, os.path.dirname(path)
            )
            path = converted

        effective_size = f.paper_size
        if job.fit_long_on_short and f.paper_size == "Long":
            effective_size = "Short"
            if pdf_utils.is_pdf_file(path):
                if f.scaled_path and os.path.exists(f.scaled_path):
                    return f.scaled_path, effective_size
                output_path = pdf_utils.scaled_pdf_path(path, "Short")
                await asyncio.to_thread(
                    pdf_utils.scale_pdf_to_paper_size,
                    path, output_path, "Short",
                )
                f.scaled_path = output_path
                self.state.save_job(job)
                return output_path, effective_size
        return path, effective_size

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
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.error("Notification task failed", exc_info=result)

    async def _notify_email(self, job: PrintJob, text: str, file_paths: Optional[list[str]]):
        try:
            result = await asyncio.to_thread(
                gmail_client.send_reply,
                self.gmail_service,
                job.reply_to_address,
                job.subject,
                job.thread_id,
                job.original_rfc_message_id,
                text,
                file_paths,
            )
            if result.internal_date_ms > 0:
                self.state.update_last_seen_internal_date(
                    job.message_id, result.internal_date_ms
                )
        except Exception:
            logger.exception(
                "Failed to send email notification for job %s", job.message_id
            )
