"""Thin wrapper around the Gmail API: authentication, listing candidate
messages, downloading attachments, and now also replying in-thread and
reading new replies for the email-based confirmation flow.

First-time setup (see README) requires a browser to grant access. On a
headless Pi, run the auth step once on a machine with a browser (or use SSH
port forwarding) and then copy the generated token file to the Pi.

NOTE: this module now requests both readonly and send scopes. If you're
upgrading from an earlier version of this bot, delete your existing
token file and re-run the auth flow so it's granted the new scope.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import make_msgid
import errno
import http.client
import logging
import os
import socket
import threading

import httplib2
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_httplib2 import AuthorizedHttp
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

# Passed to every .execute() call. The Google API client's own retry logic
# catches transient network errors (timeouts, connection resets) as well
# as retryable HTTP errors (429/5xx) and retries with exponential backoff
# -- without this, a single flaky Wi-Fi moment throws instead of retrying.
NUM_RETRIES = 3

# Socket timeout for httplib2. After a timeout the pooled connection is often
# left half-closed; retries on that same connection raise CannotSendHeader.
HTTP_TIMEOUT_SECONDS = 60

# Extra reconnect attempts after the connection is known to be poisoned
# (beyond the per-request NUM_RETRIES on a healthy connection).
RECONNECT_ATTEMPTS = 2

logger = logging.getLogger(__name__)

# httplib2 is not thread-safe. Discord runs poll_gmail and poll_email_replies
# concurrently via asyncio.to_thread on one shared service -- serialize all
# HTTP use (including reconnect) so one thread cannot close sockets another
# is still writing to (EBADF / heap corruption / ABRT).
_gmail_http_lock = threading.Lock()

# Errors that mean httplib2's connection pool is unusable until rebuilt.
_POISONED_CONNECTION_ERRORS = (
    http.client.CannotSendHeader,
    http.client.CannotSendRequest,
    http.client.ResponseNotReady,
    http.client.RemoteDisconnected,
    socket.timeout,
    TimeoutError,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
)


@dataclass
class Attachment:
    attachment_id: str
    filename: str
    mime_type: str


@dataclass
class EmailMessage:
    message_id: str
    thread_id: str
    subject: str
    sender: str                # display "From" header, for showing to the user
    reply_to_address: str      # address to actually send replies to
    rfc_message_id: str        # the Message-Id header, for In-Reply-To/References
    body_text: str
    internal_date_ms: int
    attachments: list[Attachment] = field(default_factory=list)


@dataclass
class SendReplyResult:
    rfc_message_id: str
    internal_date_ms: int


@dataclass
class ThreadReply:
    from_address: str
    body_text: str
    internal_date_ms: int


def _build_authorized_http(creds: Credentials) -> AuthorizedHttp:
    return AuthorizedHttp(creds, http=httplib2.Http(timeout=HTTP_TIMEOUT_SECONDS))


def _attach_credentials(service, creds: Credentials):
    """Stash OAuth creds on the discovery Resource so reconnect can rebuild HTTP."""
    service._printbot_credentials = creds


def get_gmail_service(credentials_file: str, token_file: str):
    """Returns an authenticated Gmail API service object, running the OAuth
    flow interactively the first time and reusing/refreshing the saved
    token afterwards."""
    creds = None

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                logger.warning(
                    "Saved Gmail token refresh failed; re-running OAuth flow."
                )
                creds = None
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file, SCOPES
            )
            # run_local_server opens a browser on the machine running this.
            # On a headless Pi, run this once on your laptop instead and
            # copy the resulting token file over (see README).
            creds = flow.run_local_server(port=0)

        os.makedirs(os.path.dirname(token_file) or ".", exist_ok=True)
        with open(token_file, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    http = _build_authorized_http(creds)
    service = build(
        "gmail", "v1", http=http, cache_discovery=False,
    )
    _attach_credentials(service, creds)
    return service


def _close_http_connections(http_obj) -> None:
    """Close pooled sockets on an AuthorizedHttp / httplib2.Http."""
    underlying = getattr(http_obj, "http", http_obj)
    connections = getattr(underlying, "connections", None)
    if not isinstance(connections, dict):
        return
    for conn in list(connections.values()):
        try:
            conn.close()
        except Exception:
            pass
    connections.clear()


def _reconnect_http(service) -> None:
    """Replace the service's AuthorizedHttp after a poisoned connection."""
    creds = getattr(service, "_printbot_credentials", None)
    if creds is None:
        raise RuntimeError(
            "Gmail service has no stored credentials; cannot reconnect HTTP."
        )

    old_http = getattr(service, "_http", None)
    if old_http is not None:
        try:
            _close_http_connections(old_http)
        except Exception:
            logger.debug("Ignoring error while closing old Gmail HTTP", exc_info=True)

    new_http = _build_authorized_http(creds)
    service._http = new_http
    logger.warning("Rebuilt Gmail HTTP connection after a poisoned/timed-out socket")


def _is_html_bad_request(error: HttpError) -> bool:
    """True for Google's HTML 400 page (often a mangled request on a dead conn)."""
    if getattr(error, "resp", None) is None or error.resp.status != 400:
        return False
    content = error.content or b""
    if isinstance(content, str):
        content = content.encode("utf-8", errors="replace")
    lowered = content[:500].lower()
    return b"<!doctype html" in lowered or b"<html" in lowered


def _is_poisoned_connection_error(error: BaseException) -> bool:
    if isinstance(error, _POISONED_CONNECTION_ERRORS):
        return True
    if isinstance(error, HttpError) and _is_html_bad_request(error):
        return True
    # One thread closed/replaced the socket while another was writing.
    if isinstance(error, OSError) and getattr(error, "errno", None) == errno.EBADF:
        return True
    message = str(error).lower()
    return (
        "timed out" in message
        or "the read operation timed out" in message
        or "record_layer_failure" in message
        or "bad file descriptor" in message
    )


def is_transient_gmail_error(error: BaseException) -> bool:
    """True for network/connection blips that should retry next poll without
    a full traceback in journalctl."""
    if _is_poisoned_connection_error(error):
        return True
    if isinstance(error, HttpError):
        status = getattr(getattr(error, "resp", None), "status", None)
        # 429/5xx are transient; HTML 400 already covered as poisoned.
        if status in (429, 500, 502, 503, 504):
            return True
    return False


def _execute(service, request):
    """Run request.execute with reconnect+retry when httplib2's connection dies.

    googleapiclient's own num_retries handles clean transient errors. After a
    timeout, the pooled connection is often half-closed and the next attempt
    raises CannotSendHeader (or a mangled HTML 400). Those need a fresh Http.

    All Gmail HTTP (including reconnect) is serialized on _gmail_http_lock
    because httplib2 is not safe across the bot's concurrent poll threads.
    """
    last_error: BaseException | None = None

    with _gmail_http_lock:
        for attempt in range(RECONNECT_ATTEMPTS + 1):
            try:
                # Keep the request's http in sync with the service after reconnects.
                request.http = service._http
                return request.execute(num_retries=NUM_RETRIES)
            except Exception as e:
                last_error = e
                if attempt >= RECONNECT_ATTEMPTS or not _is_poisoned_connection_error(e):
                    raise
                logger.warning(
                    "Gmail request failed with poisoned connection (%s); "
                    "reconnecting (attempt %d/%d)",
                    type(e).__name__,
                    attempt + 1,
                    RECONNECT_ATTEMPTS,
                )
                _reconnect_http(service)

    assert last_error is not None
    raise last_error


def get_own_email_address(service) -> str:
    profile = _execute(service, service.users().getProfile(userId="me"))
    return profile["emailAddress"].lower()


def _extract_body_text(payload: dict) -> str:
    """Walks the MIME tree and returns the first text/plain part it finds."""
    if payload.get("mimeType") == "text/plain" and "data" in payload.get("body", {}):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode(
            "utf-8", errors="replace"
        )

    for part in payload.get("parts", []):
        text = _extract_body_text(part)
        if text:
            return text

    return ""


def _extract_attachments(payload: dict) -> list[Attachment]:
    attachments = []

    def walk(part):
        filename = part.get("filename")
        body = part.get("body", {})
        if filename and body.get("attachmentId"):
            attachments.append(
                Attachment(
                    attachment_id=body["attachmentId"],
                    filename=filename,
                    mime_type=part.get("mimeType", "application/octet-stream"),
                )
            )
        for sub_part in part.get("parts", []):
            walk(sub_part)

    walk(payload)
    return attachments


def _header(headers: list[dict], name: str) -> str:
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def _extract_email_address(header_value: str) -> str:
    """Pulls a bare address out of a "Display Name <addr@x.com>" header."""
    if "<" in header_value and ">" in header_value:
        return header_value.split("<", 1)[1].split(">", 1)[0].strip().lower()
    return header_value.strip().lower()


def list_candidate_message_ids(service, query: str) -> list[str]:
    """Returns message IDs matching the search query (most recent first)."""
    message_ids = []
    request = service.users().messages().list(userId="me", q=query)
    while request is not None:
        response = _execute(service, request)
        message_ids.extend(m["id"] for m in response.get("messages", []))
        request = service.users().messages().list_next(request, response)
    return message_ids


def get_message(service, message_id: str) -> EmailMessage:
    raw = _execute(
        service,
        service.users().messages().get(
            userId="me", id=message_id, format="full"
        ),
    )

    payload = raw["payload"]
    headers = payload.get("headers", [])
    from_header = _header(headers, "From")
    reply_to_header = _header(headers, "Reply-To") or from_header

    return EmailMessage(
        message_id=message_id,
        thread_id=raw["threadId"],
        subject=_header(headers, "Subject") or "(no subject)",
        sender=from_header or "(unknown sender)",
        reply_to_address=_extract_email_address(reply_to_header),
        rfc_message_id=_header(headers, "Message-Id"),
        body_text=_extract_body_text(payload),
        internal_date_ms=int(raw.get("internalDate", "0")),
        attachments=_extract_attachments(payload),
    )


def download_attachment(
    service, message_id: str, attachment: Attachment, dest_path: str
) -> str:
    """Downloads an attachment to dest_path and returns that path."""
    data = _execute(
        service,
        service.users()
        .messages()
        .attachments()
        .get(userId="me", messageId=message_id, id=attachment.attachment_id),
    )
    file_bytes = base64.urlsafe_b64decode(data["data"])
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(file_bytes)
    return dest_path


def send_reply(
    service,
    to_address: str,
    subject: str,
    thread_id: str,
    in_reply_to_rfc_id: str,
    body_text: str,
    attachment_paths: list[str] | None = None,
) -> SendReplyResult:
    """Sends a reply within an existing Gmail thread. Returns the RFC
    Message-Id and the sent message's Gmail internalDate (for reply
    watermarking)."""
    msg = MIMEMultipart()
    msg["To"] = to_address
    msg["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    own_rfc_id = make_msgid()
    msg["Message-Id"] = own_rfc_id
    if in_reply_to_rfc_id:
        msg["In-Reply-To"] = in_reply_to_rfc_id
        msg["References"] = in_reply_to_rfc_id

    msg.attach(MIMEText(body_text, "plain"))

    for path in attachment_paths or []:
        with open(path, "rb") as f:
            file_bytes = f.read()
        part = MIMEApplication(file_bytes, _subtype="pdf")
        part.add_header(
            "Content-Disposition", "attachment", filename=os.path.basename(path)
        )
        msg.attach(part)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    # Note: unlike the read-only calls above, retrying a send isn't
    # perfectly safe -- if the message actually went through but the
    # response was lost to the same kind of network hiccup, a retry could
    # send a duplicate confirmation email. That's a minor annoyance
    # compared to silently dropping a confirmation/result message, so it
    # still retries here.
    sent = _execute(
        service,
        service.users().messages().send(
            userId="me", body={"raw": raw, "threadId": thread_id}
        ),
    )
    meta = _execute(
        service,
        service.users().messages().get(
            userId="me", id=sent["id"], format="minimal"
        ),
    )
    internal_date_ms = int(meta.get("internalDate", "0"))

    return SendReplyResult(
        rfc_message_id=own_rfc_id,
        internal_date_ms=internal_date_ms,
    )


def list_new_thread_replies(
    service, thread_id: str, since_internal_date_ms: int
) -> list[ThreadReply]:
    """Returns every message in the thread newer than
    since_internal_date_ms, oldest first. Used to detect replies that
    arrived after the bot's own confirmation-ask email."""
    thread = _execute(
        service,
        service.users().threads().get(
            userId="me", id=thread_id, format="full"
        ),
    )

    replies = []
    for message in thread.get("messages", []):
        internal_date = int(message.get("internalDate", "0"))
        if internal_date <= since_internal_date_ms:
            continue
        headers = message["payload"].get("headers", [])
        from_address = _extract_email_address(_header(headers, "From"))
        body_text = _extract_body_text(message["payload"])
        replies.append(ThreadReply(from_address, body_text, internal_date))

    replies.sort(key=lambda r: r.internal_date_ms)
    return replies
