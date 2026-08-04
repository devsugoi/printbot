"""The Discord side of the bot, plus the two background polling loops:

  - poll_gmail: finds NEW emails that might be print requests.
  - poll_email_replies: watches already-asked jobs' threads for replies
    that approve, cancel, or reprint them.

All the "what does approve/cancel actually do" logic lives in
confirmation.py; this module's job is to build the job records, build the
Discord UI, and decide whether an incoming email reply is from someone
allowed to approve things.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Optional

import discord
from discord.ext import commands, tasks

from . import gmail_client, pdf_utils
from .ai_classifier import AllKeysExhaustedError, GeminiClassifier
from .config import AppConfig
from .confirmation import ConfirmationManager
from .state import (
    STATUS_AWAITING_CONFIRMATION,
    STATUS_CONFIRMED,
    STATUS_FAILED,
    STATUS_PRINTED,
    STATUS_PRINTING,
    PrintFile,
    PrintJob,
    StateStore,
)

logger = logging.getLogger(__name__)


class PrintBot(commands.Bot):
    def __init__(self, config: AppConfig, state: StateStore):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=config.discord.command_prefix, intents=intents)

        self.app_config = config
        self.state = state
        self.classifier = GeminiClassifier(
            api_keys=config.gemini.api_keys, models=config.gemini.models
        )
        self.gmail_service = None       # set in setup_hook (blocking call)
        self.owner_email = ""           # set in setup_hook
        self.confirmation: Optional[ConfirmationManager] = None
        self._notify_channel: discord.abc.Messageable | None = None
        self._announced_online = False

        self._register_commands()

    # -- lifecycle ------------------------------------------------------

    async def setup_hook(self):
        # OAuth/token refresh and the profile lookup are blocking I/O --
        # keep them off the event loop.
        self.gmail_service = await asyncio.to_thread(
            gmail_client.get_gmail_service,
            self.app_config.gmail.credentials_file,
            self.app_config.gmail.token_file,
        )
        self.owner_email = await asyncio.to_thread(
            gmail_client.get_own_email_address, self.gmail_service
        )

        self.confirmation = ConfirmationManager(
            self.app_config, self.state, self.gmail_service
        )
        self.confirmation.on_notify_discord = self._post_notification

        self._recover_interrupted_jobs()
        self._reregister_persistent_views()

        interval = self.app_config.gmail.poll_interval_seconds
        self.poll_gmail.change_interval(seconds=interval)
        self.poll_gmail.start()
        self.poll_email_replies.change_interval(seconds=interval)
        self.poll_email_replies.start()

    async def _ensure_notify_channel(self) -> discord.abc.Messageable | None:
        if self._notify_channel is not None:
            return self._notify_channel
        channel = self.get_channel(self.app_config.discord.channel_id)
        if channel is None:
            try:
                channel = await self.fetch_channel(self.app_config.discord.channel_id)
            except discord.DiscordException:
                logger.exception("Could not resolve Discord notify channel")
                return None
        self._notify_channel = channel
        return channel

    def _recover_interrupted_jobs(self):
        """If the bot crashed or was restarted while a job was mid-print
        (status "confirmed" or "printing"), that job would otherwise be
        stuck forever -- neither channel currently offers a way to act on
        those transient statuses. Treat them as failed so the normal
        Reprint path (button + email reply) picks them back up."""
        for job in self.state.all_jobs():
            if job.status in (STATUS_CONFIRMED, STATUS_PRINTING):
                logger.warning(
                    "Job %s was interrupted mid-print (status=%s) by a "
                    "restart; marking failed so it can be reprinted.",
                    job.message_id, job.status,
                )
                job.status = STATUS_FAILED
                job.last_error = "Interrupted by a bot restart before printing finished."
                self.state.save_job(job)

    def _reregister_persistent_views(self):
        """Discord buttons only route to a view object that's currently
        registered in this process. Without this, restarting the bot
        would leave every previously-sent "Print"/"Cancel"/"Reprint"
        button on old messages non-functional -- clicking one shows
        Discord's generic "This interaction failed" with nothing logged
        here, since the interaction never reaches our code. Re-adding a
        matching view (same custom_ids, see ConfirmView/ActionView) for
        every job that could still have a live button fixes that."""
        owner_id = self.app_config.discord.user_id
        jobs = self.state.jobs_awaiting_email_replies(
            self.app_config.storage.processed_email_retention_days
        )
        for job in jobs:
            if job.status == STATUS_AWAITING_CONFIRMATION:
                self.add_view(ConfirmView(
                    self.confirmation, job.message_id, owner_id,
                    has_long_paper=job.has_long_paper_pending(),
                ))
            elif job.status == STATUS_PRINTED:
                self.add_view(ActionView(
                    self.confirmation, job.message_id, owner_id,
                    label="Print again", style=discord.ButtonStyle.primary,
                    default_copies=job.copies,
                ))
            elif job.status == STATUS_FAILED:
                self.add_view(ActionView(
                    self.confirmation, job.message_id, owner_id,
                    label="Reprint", style=discord.ButtonStyle.danger,
                    default_copies=job.copies,
                ))
        logger.info("Re-registered Discord views for %d job(s)", len(jobs))

    async def on_ready(self):
        logger.info("Logged in as %s (owner mailbox: %s)", self.user, self.owner_email)
        channel = await self._ensure_notify_channel()
        if channel is None:
            logger.error(
                "Discord channel %s not found; notifications will be dropped.",
                self.app_config.discord.channel_id,
            )
            return
        if not self._announced_online:
            self._announced_online = True
            await channel.send(
                "🖨️ Print bot is online — watching Gmail for print requests and "
                "this channel + email for confirmations."
            )

    def _is_owner(self, user: discord.abc.User) -> bool:
        return user.id == self.app_config.discord.user_id

    def _is_reply_authorized(self, from_address: str) -> bool:
        """Decides whether an email reply counts toward approving/
        cancelling/reprinting a job. Your own address always counts; see
        config.example.yaml for the whitelist / open-approval options."""
        if from_address == self.owner_email:
            return True

        whitelist = self.app_config.gmail.approved_reply_senders
        if whitelist:
            return from_address in whitelist

        return self.app_config.gmail.allow_non_owner_email_approval

    # -- Gmail polling: detect NEW print-request emails --------------------

    @tasks.loop(seconds=60)
    async def poll_gmail(self):
        try:
            await self._poll_gmail_once()
        except Exception as e:
            if gmail_client.is_transient_gmail_error(e):
                logger.warning(
                    "Transient error while polling Gmail for new emails: %s: %s",
                    type(e).__name__, e,
                )
            else:
                logger.exception("Error while polling Gmail for new emails")

    @poll_gmail.before_loop
    async def _before_poll_gmail(self):
        await self.wait_until_ready()

    async def _poll_gmail_once(self):
        message_ids = await asyncio.to_thread(
            gmail_client.list_candidate_message_ids,
            self.gmail_service,
            self.app_config.gmail.search_query,
        )

        for message_id in message_ids:
            if self.state.is_processed(message_id):
                continue
            try:
                await self._handle_candidate_message(message_id)
            except Exception as e:
                # Leave it unmarked so a transient failure (network, etc.)
                # gets retried next poll instead of being silently
                # skipped forever -- and don't let it stop the rest of
                # this batch from being checked.
                if gmail_client.is_transient_gmail_error(e):
                    logger.warning(
                        "Transient failure handling candidate %s -- will retry "
                        "next poll: %s: %s",
                        message_id, type(e).__name__, e,
                    )
                else:
                    logger.exception(
                        "Failed to handle candidate message %s -- will retry next poll",
                        message_id,
                    )
                continue
            # Only mark processed once handling actually completed (found
            # a print request and asked about it, or decided it wasn't one).
            self.state.mark_processed(message_id)

    async def _handle_candidate_message(self, message_id: str):
        email = await asyncio.to_thread(
            gmail_client.get_message, self.gmail_service, message_id
        )

        if not email.attachments:
            return

        try:
            result = await asyncio.to_thread(
                self.classifier.classify,
                email.subject,
                email.sender,
                email.body_text,
                [a.filename for a in email.attachments],
            )
        except AllKeysExhaustedError as e:
            logger.error("Gemini classification failed for %s: %s", message_id, e)
            channel = await self._ensure_notify_channel()
            if channel is not None:
                await channel.send(
                    f"⚠️ Couldn't check an email (\"{email.subject}\") because all "
                    f"Gemini API keys/models are currently unavailable. It will "
                    f"be skipped. Error: {e}"
                )
            return

        if not result.is_print_request:
            logger.info("Not a print request: %s (%s)", email.subject, result.reason)
            return

        job = await self._prepare_job(email, result.paper_size)
        self.state.save_job(job)
        await self._send_initial_ask(job, result.reason)

    async def _prepare_job(
        self, email: gmail_client.EmailMessage, ai_paper_size: Optional[str]
    ) -> PrintJob:
        job_dir = os.path.join(self.app_config.storage.jobs_dir, email.message_id)
        os.makedirs(job_dir, exist_ok=True)
        supported = self.app_config.printer.supported_paper_sizes
        default = self.app_config.printer.default_paper_size

        downloaded_paths = []
        used_filenames: set[str] = set()
        for attachment in email.attachments:
            safe_name = _safe_attachment_filename(attachment.filename, used_filenames)
            dest = os.path.join(job_dir, safe_name)
            await asyncio.to_thread(
                gmail_client.download_attachment,
                self.gmail_service, email.message_id, attachment, dest,
            )
            downloaded_paths.append(dest)

        image_paths = [p for p in downloaded_paths if pdf_utils.is_image_file(p)]
        pdf_paths = [p for p in downloaded_paths if pdf_utils.is_pdf_file(p)]
        other_paths = [
            p for p in downloaded_paths if p not in image_paths and p not in pdf_paths
        ]

        files: list[PrintFile] = []

        for p in pdf_paths:
            size = ai_paper_size or pdf_utils.detect_pdf_paper_size(p, supported, default)
            files.append(PrintFile(path=p, paper_size=size, is_generated=False))

        for p in other_paths:
            if pdf_utils.is_office_file(p):
                # Office documents (.docx, .xlsx, ...) can't be printed by
                # CUPS directly -- convert to PDF with LibreOffice. The
                # converted PDF is marked is_generated so it's attached to
                # the confirmation ask as a preview.
                try:
                    converted = await asyncio.to_thread(
                        pdf_utils.office_to_pdf, p, job_dir
                    )
                except pdf_utils.OfficeConversionError:
                    logger.exception(
                        "Failed to convert %s to PDF at prepare time; "
                        "queueing the original file (conversion will be "
                        "retried at print time)", p,
                    )
                    files.append(
                        PrintFile(
                            path=p,
                            paper_size=ai_paper_size or default,
                            is_generated=False,
                        )
                    )
                    continue
                size = ai_paper_size or pdf_utils.detect_pdf_paper_size(
                    converted, supported, default
                )
                files.append(
                    PrintFile(path=converted, paper_size=size, is_generated=True)
                )
                continue

            # Anything else -- best effort, sent to lp as-is. See README
            # for the caveat about non-PDF print reliability.
            files.append(
                PrintFile(path=p, paper_size=ai_paper_size or default, is_generated=False)
            )

        if image_paths:
            # Images are always combined into a single short-bond-paper
            # PDF, one image per page, scaled to fill the page.
            _, size_pt = pdf_utils.resolve_paper_size("Short", default)
            combined_pdf = os.path.join(job_dir, "combined_images.pdf")
            await asyncio.to_thread(
                pdf_utils.images_to_pdf, image_paths, combined_pdf, size_pt
            )
            files.append(PrintFile(path=combined_pdf, paper_size="Short", is_generated=True))

        return PrintJob(
            message_id=email.message_id,
            thread_id=email.thread_id,
            subject=email.subject,
            sender=email.sender,
            reply_to_address=email.reply_to_address,
            owner_email=self.owner_email,
            original_rfc_message_id=email.rfc_message_id,
            files=files,
        )

    async def _send_initial_ask(self, job: PrintJob, ai_reason: str):
        file_list = ", ".join(os.path.basename(f.path) for f in job.files)
        warning = self._paper_warning(job)

        discord_text = (
            f"🖨️ **Print request detected**\n"
            f"**From:** {job.sender}\n"
            f"**Subject:** {job.subject}\n"
            f"**Files:** {file_list}\n"
            f"**Why:** {ai_reason}"
            + (f"\n\n{warning}" if warning else "")
        )
        email_text = (
            f"I think this email is asking to print the attached file(s): "
            f"{file_list}.\n\n"
            f'Reply to this thread with something like "yes, 2 copies" to '
            f'confirm, or "no" to cancel. You can also confirm on Discord.'
            + (f"\n\n{warning}" if warning else "")
        )

        # Only attachments the bot itself generated (e.g. images combined
        # into a PDF) get sent back for you to preview -- files you
        # already sent yourself don't need to be echoed back.
        preview_paths = [f.path for f in job.generated_files()]

        await self.confirmation.notify_both(
            job, discord_text, email_text, file_paths=preview_paths
        )

    @staticmethod
    def _paper_warning(job: PrintJob) -> str:
        sizes_needed = {f.paper_size for f in job.files}
        if "Long" in sizes_needed:
            return (
                '⚠️ This needs **long bond paper** (8.5"x14") at some point. '
                "The tray usually has short bond paper loaded — have long "
                "bond paper ready to swap in. Or click **Print on short bond** / "
                'reply "use short bond" to scale it onto the paper already loaded.'
            )
        return ""

    # -- email polling: watch for replies that approve/cancel/reprint ------

    @tasks.loop(seconds=60)
    async def poll_email_replies(self):
        try:
            await self._poll_email_replies_once()
        except Exception as e:
            if gmail_client.is_transient_gmail_error(e):
                logger.warning(
                    "Transient error while polling email replies: %s: %s",
                    type(e).__name__, e,
                )
            else:
                logger.exception("Error while polling email replies")

    @poll_email_replies.before_loop
    async def _before_poll_email_replies(self):
        await self.wait_until_ready()

    async def _poll_email_replies_once(self):
        jobs = self.state.jobs_awaiting_email_replies(
            self.app_config.storage.processed_email_retention_days
        )

        for job in jobs:
            try:
                replies = await asyncio.to_thread(
                    gmail_client.list_new_thread_replies,
                    self.gmail_service, job.thread_id, job.last_seen_internal_date_ms,
                )
            except Exception as e:
                # A transient error (e.g. a network hiccup) fetching one
                # job's thread shouldn't stop every other job from being
                # checked this cycle -- log it and move on; it'll be
                # retried next poll.
                if gmail_client.is_transient_gmail_error(e):
                    logger.warning(
                        "Transient failure fetching replies for job %s -- will "
                        "retry next poll: %s: %s",
                        job.message_id, type(e).__name__, e,
                    )
                else:
                    logger.exception(
                        "Failed to fetch replies for job %s -- will retry next poll",
                        job.message_id,
                    )
                continue

            for reply in replies:
                if not self._is_reply_authorized(reply.from_address):
                    logger.info(
                        "Ignoring reply from unauthorized address %s on job %s",
                        reply.from_address, job.message_id,
                    )
                    continue

                try:
                    decision = await asyncio.to_thread(
                        self.classifier.classify_reply, reply.body_text
                    )
                except AllKeysExhaustedError as e:
                    logger.error(
                        "Could not classify email reply for job %s: %s",
                        job.message_id, e,
                    )
                    break

                # Re-fetch in case an earlier reply in this same batch (or
                # a Discord click) already changed the job.
                current = self.state.get_job(job.message_id) or job
                current.last_seen_internal_date_ms = max(
                    current.last_seen_internal_date_ms, reply.internal_date_ms
                )
                self.state.save_job(current)

                # If the job already looks finished/failed by the time we
                # notice this reply, treat it as a deliberate "print
                # again"/"reprint" rather than a race against the original
                # ask -- handle_approval makes the authoritative call once
                # it holds the lock either way.
                is_explicit_retry = current.status != STATUS_AWAITING_CONFIRMATION

                if decision.decision == "approve":
                    await self.confirmation.handle_approval(
                        job.message_id, source="email",
                        actor=reply.from_address, copies=decision.copies,
                        is_explicit_retry=is_explicit_retry,
                        fit_long_on_short=decision.fit_on_short,
                    )
                elif decision.decision == "cancel":
                    await self.confirmation.handle_cancel(
                        job.message_id, source="email", actor=reply.from_address,
                    )
                # "unclear" -> leave it; wait for a clearer reply.

    # -- Discord notification callback (called by confirmation.py) --------

    async def _post_notification(
        self, job: PrintJob, text: str, file_paths: Optional[list[str]]
    ):
        channel = await self._ensure_notify_channel()
        if channel is None:
            return

        view = None
        owner_id = self.app_config.discord.user_id
        if job.status == STATUS_AWAITING_CONFIRMATION:
            view = ConfirmView(
                self.confirmation, job.message_id, owner_id,
                has_long_paper=job.has_long_paper_pending(),
            )
        elif job.status == STATUS_PRINTED:
            view = ActionView(
                self.confirmation, job.message_id, owner_id,
                label="Print again", style=discord.ButtonStyle.primary,
                default_copies=job.copies,
            )
        elif job.status == STATUS_FAILED:
            view = ActionView(
                self.confirmation, job.message_id, owner_id,
                label="Reprint", style=discord.ButtonStyle.danger,
                default_copies=job.copies,
            )

        send_kwargs = {"content": text}
        if view is not None:
            send_kwargs["view"] = view
        existing_files = [p for p in (file_paths or []) if os.path.exists(p)]
        if existing_files:
            send_kwargs["files"] = [discord.File(p) for p in existing_files]

        await channel.send(**send_kwargs)

    # -- text commands ------------------------------------------------

    def _register_commands(self):
        @self.command(name="status")
        async def status_cmd(ctx: commands.Context):
            if not self._is_owner(ctx.author):
                return
            jobs = sorted(self.state.all_jobs(), key=lambda j: j.created_at, reverse=True)[:10]
            if not jobs:
                await ctx.send("No print jobs recorded yet.")
                return
            lines = [
                f"`{j.message_id}` [{j.status}] {j.subject} ({j.copies}x)"
                for j in jobs
            ]
            text = "**Recent jobs:**\n" + "\n".join(lines)
            if len(text) > 2000:
                text = text[:1997] + "..."
            await ctx.send(text)

        @self.command(name="reprint")
        async def reprint_cmd(ctx: commands.Context, message_id: str, copies: Optional[int] = None):
            if not self._is_owner(ctx.author):
                return
            job = self.state.get_job(message_id)
            if job is None:
                await ctx.send(f"No job found for id `{message_id}`.")
                return
            await ctx.send(f"Working on **{job.subject}**...")
            accepted = await self.confirmation.handle_approval(
                message_id, source="discord", actor=str(ctx.author),
                copies=copies, is_explicit_retry=True,
            )
            if not accepted:
                await ctx.send(
                    f"Cannot reprint job `{message_id}` (status: `{job.status}`)."
                )


class CopiesModal(discord.ui.Modal):
    """Pops up when Print / Print again / Reprint is clicked, asking how
    many copies. Leaving it blank keeps the previous count (or defaults to
    1 for a brand new job)."""

    def __init__(
        self, confirmation: ConfirmationManager, message_id: str,
        default_copies: int = 1, is_explicit_retry: bool = False,
        fit_long_on_short: bool = False,
        title: str = "How many copies?",
    ):
        super().__init__(title=title)
        self.confirmation = confirmation
        self.message_id = message_id
        self.is_explicit_retry = is_explicit_retry
        self.fit_long_on_short = fit_long_on_short
        self.copies_input = discord.ui.TextInput(
            label="Number of copies",
            default=str(default_copies),
            required=False,
            max_length=3,
        )
        self.add_item(self.copies_input)

    async def on_submit(self, interaction: discord.Interaction):
        raw = (self.copies_input.value or "").strip()
        copies = int(raw) if raw.isdigit() and int(raw) > 0 else None

        await interaction.response.send_message(
            "Got it — working on it now.", ephemeral=True
        )
        await self.confirmation.handle_approval(
            self.message_id, source="discord", actor=str(interaction.user),
            copies=copies, is_explicit_retry=self.is_explicit_retry,
            fit_long_on_short=self.fit_long_on_short,
        )


class ConfirmView(discord.ui.View):
    def __init__(
        self, confirmation: ConfirmationManager, message_id: str,
        owner_id: int, has_long_paper: bool = False,
    ):
        super().__init__(timeout=None)
        self.confirmation = confirmation
        self.message_id = message_id
        self.owner_id = owner_id

        print_btn = discord.ui.Button(
            label="Print", style=discord.ButtonStyle.success, emoji="🖨️",
            custom_id=f"printbot:confirm:{message_id}",
        )
        print_btn.callback = self._on_print
        self.add_item(print_btn)

        if has_long_paper:
            short_btn = discord.ui.Button(
                label="Print on short bond", style=discord.ButtonStyle.primary,
                emoji="📄",
                custom_id=f"printbot:confirm-short:{message_id}",
            )
            short_btn.callback = self._on_print_short
            self.add_item(short_btn)

        cancel_btn = discord.ui.Button(
            label="Cancel", style=discord.ButtonStyle.danger, emoji="🚫",
            custom_id=f"printbot:cancel:{message_id}",
        )
        cancel_btn.callback = self._on_cancel
        self.add_item(cancel_btn)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "This isn't your print request.", ephemeral=True
            )
            return False
        return True

    async def _on_print(self, interaction: discord.Interaction):
        await interaction.response.send_modal(
            CopiesModal(self.confirmation, self.message_id)
        )

    async def _on_print_short(self, interaction: discord.Interaction):
        await interaction.response.send_modal(
            CopiesModal(
                self.confirmation, self.message_id,
                fit_long_on_short=True,
                title="Print on short bond (scaled to fit)",
            )
        )

    async def _on_cancel(self, interaction: discord.Interaction):
        await interaction.response.defer()
        await self.confirmation.handle_cancel(
            self.message_id, source="discord", actor=str(interaction.user)
        )


class ActionView(discord.ui.View):
    """A single-button view used for "Print again" (after success) and
    "Reprint" (after failure) -- both just re-enter the same approval
    path with a fresh copies prompt."""

    def __init__(
        self, confirmation: ConfirmationManager, message_id: str, owner_id: int,
        label: str, style: discord.ButtonStyle, default_copies: int = 1,
    ):
        super().__init__(timeout=None)
        self.confirmation = confirmation
        self.message_id = message_id
        self.owner_id = owner_id
        self.default_copies = default_copies

        button = discord.ui.Button(
            label=label, style=style, emoji="🔁",
            custom_id=f"printbot:retry:{message_id}",
        )
        button.callback = self._on_click
        self.add_item(button)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "This isn't your print job.", ephemeral=True
            )
            return False
        return True

    async def _on_click(self, interaction: discord.Interaction):
        await interaction.response.send_modal(
            CopiesModal(
                self.confirmation, self.message_id,
                default_copies=self.default_copies, is_explicit_retry=True,
            )
        )


def _safe_attachment_filename(filename: str, used_names: set[str]) -> str:
    """Sanitize an email attachment filename for safe local storage."""
    base = os.path.basename(filename or "attachment")
    base = re.sub(r"[^\w.\- ]", "_", base).strip() or "attachment"
    if base in (".", ".."):
        base = "attachment"

    name, ext = os.path.splitext(base)
    candidate = base
    counter = 1
    while candidate.lower() in used_names:
        candidate = f"{name}-{counter}{ext}"
        counter += 1
    used_names.add(candidate.lower())
    return candidate
