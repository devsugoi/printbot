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
import os

import httplib2
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]


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
class ThreadReply:
    from_address: str
    body_text: str
    internal_date_ms: int


def get_gmail_service(credentials_file: str, token_file: str):
    """Returns an authenticated Gmail API service object, running the OAuth
    flow interactively the first time and reusing/refreshing the saved
    token afterwards."""
    creds = None

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
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

    http = httplib2.Http(timeout=30)
    return build("gmail", "v1", credentials=creds, http=http)


def get_own_email_address(service) -> str:
    profile = service.users().getProfile(userId="me").execute()
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
        response = request.execute()
        message_ids.extend(m["id"] for m in response.get("messages", []))
        request = service.users().messages().list_next(request, response)
    return message_ids


def get_message(service, message_id: str) -> EmailMessage:
    raw = service.users().messages().get(
        userId="me", id=message_id, format="full"
    ).execute()

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
    data = (
        service.users()
        .messages()
        .attachments()
        .get(userId="me", messageId=message_id, id=attachment.attachment_id)
        .execute()
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
) -> str:
    """Sends a reply within an existing Gmail thread. Returns the Message-Id
    (RFC header, not the Gmail id) of the message we just sent, so it can be
    used as In-Reply-To for any further reply in the same job."""
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
    service.users().messages().send(
        userId="me", body={"raw": raw, "threadId": thread_id}
    ).execute()

    return own_rfc_id


def list_new_thread_replies(
    service, thread_id: str, since_internal_date_ms: int
) -> list[ThreadReply]:
    """Returns every message in the thread newer than
    since_internal_date_ms, oldest first. Used to detect replies that
    arrived after the bot's own confirmation-ask email."""
    thread = service.users().threads().get(
        userId="me", id=thread_id, format="full"
    ).execute()

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
