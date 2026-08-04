"""Uses Gemini (free tier) for two jobs:
  1. Deciding whether an email is asking for an attachment to be printed
     (and guessing a paper size if the email hints at one).
  2. Interpreting short reply emails ("yes, 2 copies" / "cancel that") sent
     in response to the bot's own confirmation-ask messages.

Uses the current `google-genai` SDK (the older `google-generativeai`
package is deprecated).

Handles two kinds of fallback, shared by both jobs above:
  1. Model fallback: if a model is unavailable/overloaded, try the next
     model configured for the same API key.
  2. Key fallback: if a key is out of quota (or invalid), move to the next
     API key and retry the model list from the top.

This keeps the bot working even on the Gemini free tier's fairly low
per-key rate limits, as long as you provide a few keys from different
Google accounts/projects.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import errors

logger = logging.getLogger(__name__)

# HTTP status codes that mean "this specific model is unusable right now"
# -> try the next model before giving up on the current key.
# 404 = model not found/decommissioned, 400 = bad request for this model,
# 503 = model temporarily overloaded.
MODEL_LEVEL_STATUS_CODES = {400, 404, 503}

# HTTP status codes that mean "this key is unusable right now" -> move to
# the next key. 429 = quota/rate limit exceeded, 403 = key lacks access.
KEY_LEVEL_STATUS_CODES = {403, 429}

CLASSIFY_PROMPT_TEMPLATE = """You are a filter for an email-to-printer bot. Decide \
whether the email below is a request from the owner to print an attachment, \
and which paper size it should be printed on.

The printer is only loaded with two paper choices:
- "Short" = short bond paper (8.5 x 11 in, i.e. US Letter)
- "Long"  = long bond paper (8.5 x 14 in, i.e. US Legal / folio)

Respond with ONLY a JSON object (no markdown, no extra text) matching this \
exact shape:
{{
  "is_print_request": true or false,
  "paper_size": "Short" or "Long" or null,
  "reason": a short (<20 words) explanation of your decision
}}

Rules:
- is_print_request should be true only if the email is clearly asking for \
one or more of its own attachments to be printed.
- If the email or attachment names explicitly mention a paper size (e.g. \
"legal size", "long bond paper", "letter size", "8.5x14", "folio"), map it \
to "Short" or "Long" accordingly.
- If the attachments are plain images (photos, screenshots), ignore this \
field -- images are always printed on short bond paper regardless of what \
you return here.
- If no size is mentioned and the attachment isn't a plain image, use your \
best judgement from context (document type, filenames -- e.g. legal \
contracts/forms are often long bond paper, general letters/reports are \
usually short bond paper). Only return null if you genuinely have no basis \
to guess.

Email subject: {subject}
Email sender: {sender}
Attachment filenames: {attachments}
Email body:
---
{body}
---
"""

REPLY_PROMPT_TEMPLATE = """You are interpreting a short reply email sent in \
response to a print-confirmation request from a bot. Decide the reply's \
intent.

Respond with ONLY a JSON object (no markdown, no extra text) matching this \
exact shape:
{{
  "decision": "approve" or "cancel" or "unclear",
  "copies": an integer number of copies if one is mentioned, or null
}}

Rules:
- "approve" means the person wants the print job to proceed (e.g. "yes", \
"go ahead", "print it", "print again", "reprint", "do it").
- "cancel" means the person wants it NOT printed, or stopped (e.g. "no", \
"cancel", "stop", "don't print that").
- "unclear" if the reply doesn't clearly indicate either.
- copies should be null unless a specific number of copies is mentioned.

Reply text:
---
{reply_text}
---
"""


@dataclass
class ClassificationResult:
    is_print_request: bool
    paper_size: Optional[str]
    reason: str


@dataclass
class ReplyDecision:
    decision: str  # "approve" | "cancel" | "unclear"
    copies: Optional[int]


class AllKeysExhaustedError(RuntimeError):
    """Raised when every API key / model combination failed."""


class GeminiClassifier:
    def __init__(self, api_keys: list[str], models: list[str]):
        if not api_keys:
            raise ValueError("At least one Gemini API key is required.")
        if not models:
            raise ValueError("At least one Gemini model name is required.")
        self.api_keys = api_keys
        self.models = models

    def classify(
        self, subject: str, sender: str, body: str, attachment_names: list[str]
    ) -> ClassificationResult:
        prompt = CLASSIFY_PROMPT_TEMPLATE.format(
            subject=subject,
            sender=sender,
            attachments=", ".join(attachment_names) or "(none)",
            body=body[:4000],  # keep prompts small; free tier has token limits
        )
        raw_text = self._generate_json(prompt)
        return self._parse_classification(raw_text)

    def classify_reply(self, reply_text: str) -> ReplyDecision:
        prompt = REPLY_PROMPT_TEMPLATE.format(reply_text=reply_text[:2000])
        raw_text = self._generate_json(prompt)
        return self._parse_reply(raw_text)

    # -- shared fallback machinery ------------------------------------------

    def _generate_json(self, prompt: str) -> str:
        """Tries every (key, model) combination in order, returning the raw
        text of the first successful response. Raises AllKeysExhaustedError
        if nothing works."""
        last_error: Optional[Exception] = None

        for key_index, api_key in enumerate(self.api_keys):
            client = genai.Client(api_key=api_key)

            for model_name in self.models:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={"response_mime_type": "application/json"},
                    )
                    return response.text

                except errors.APIError as e:
                    code = getattr(e, "code", None)

                    if code in KEY_LEVEL_STATUS_CODES:
                        logger.warning(
                            "Gemini API key #%d rejected/exhausted "
                            "(HTTP %s): %s. Trying next key.",
                            key_index + 1, code, e,
                        )
                        last_error = e
                        break  # stop trying models for this key

                    logger.warning(
                        "Gemini model '%s' unavailable (key #%d, HTTP %s): "
                        "%s. Trying next model.",
                        model_name, key_index + 1, code, e,
                    )
                    last_error = e
                    continue

        raise AllKeysExhaustedError(
            f"All Gemini API keys/models failed. Last error: {last_error}"
        )

    @staticmethod
    def _parse_classification(raw_text: str) -> ClassificationResult:
        try:
            data = json.loads(raw_text)
            return ClassificationResult(
                is_print_request=bool(data.get("is_print_request", False)),
                paper_size=data.get("paper_size") or None,
                reason=str(data.get("reason", "")),
            )
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            logger.error("Could not parse Gemini response as JSON: %r", raw_text)
            # Fail safe: treat unparsable responses as "not a print request"
            # rather than risk mis-printing something.
            return ClassificationResult(
                is_print_request=False,
                paper_size=None,
                reason=f"Failed to parse model response: {e}",
            )

    @staticmethod
    def _parse_reply(raw_text: str) -> ReplyDecision:
        try:
            data = json.loads(raw_text)
            copies = data.get("copies")
            return ReplyDecision(
                decision=str(data.get("decision", "unclear")),
                copies=int(copies) if copies is not None else None,
            )
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
            logger.error(
                "Could not parse Gemini reply response as JSON: %r (%s)", raw_text, e
            )
            return ReplyDecision(decision="unclear", copies=None)
