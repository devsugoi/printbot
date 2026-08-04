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
import re
from dataclasses import dataclass
from typing import Any, Optional

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

CLASSIFY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_print_request": {"type": "boolean"},
        "paper_size": {"type": "string", "nullable": True},
        "reason": {"type": "string"},
    },
    "required": ["is_print_request", "reason"],
}

REPLY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approve", "cancel", "unclear"],
        },
        "copies": {"type": "integer", "nullable": True},
        "fit_on_short": {"type": "boolean", "nullable": True},
    },
    "required": ["decision"],
}

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
  "reason": a short (<15 words) explanation of your decision
}}

Rules:
- is_print_request should be true only if the email is clearly asking for \
one or more of its own attachments to be printed.
- The reason must be under 15 words and must not contain double quotes.
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
  "copies": an integer number of copies if one is mentioned, or null,
  "fit_on_short": true or false or null
}}

Rules:
- "approve" means the person wants the print job to proceed (e.g. "yes", \
"go ahead", "print it", "print again", "reprint", "do it").
- "cancel" means the person wants it NOT printed, or stopped (e.g. "no", \
"cancel", "stop", "don't print that").
- "unclear" if the reply doesn't clearly indicate either.
- copies should be null unless a specific number of copies is mentioned.
- fit_on_short should be true only if the person explicitly wants to print \
on short/letter bond paper instead of swapping in long bond paper (e.g. \
"use short bond", "print on letter", "fit on short", "don't swap paper"). \
Return false or null for a normal approval.

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
    fit_on_short: Optional[bool] = None


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
        data = self._generate_json(prompt, CLASSIFY_RESPONSE_SCHEMA)
        return self._classification_from_data(data)

    def classify_reply(self, reply_text: str) -> ReplyDecision:
        prompt = REPLY_PROMPT_TEMPLATE.format(reply_text=reply_text[:2000])
        data = self._generate_json(prompt, REPLY_RESPONSE_SCHEMA)
        return self._reply_from_data(data)

    # -- shared fallback machinery ------------------------------------------

    def _generate_json(self, prompt: str, response_schema: dict) -> dict:
        """Tries every (key, model) combination in order, returning the
        parsed JSON object from the first successful response. Raises
        AllKeysExhaustedError if nothing works."""
        last_error: Optional[Exception] = None

        for key_index, api_key in enumerate(self.api_keys):
            client = genai.Client(api_key=api_key)
            skip_key = False

            for model_name in self.models:
                if skip_key:
                    break
                for attempt in range(2):
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config={
                                "response_mime_type": "application/json",
                                "response_schema": response_schema,
                                "max_output_tokens": 256,
                            },
                        )
                        data, raw_text, finish_reason = self._extract_response_data(
                            response
                        )

                        if data is not None and not self._was_truncated(finish_reason):
                            return data

                        if attempt == 0:
                            logger.warning(
                                "Gemini response unusable (model=%s, key #%d, "
                                "finish_reason=%s, attempt=%d). Retrying.",
                                model_name,
                                key_index + 1,
                                finish_reason,
                                attempt + 1,
                            )
                            continue

                        if data is not None:
                            logger.warning(
                                "Using Gemini response after retry despite "
                                "finish_reason=%s (model=%s).",
                                finish_reason,
                                model_name,
                            )
                            return data

                        logger.warning(
                            "Gemini response unparseable after retry "
                            "(model=%s, key #%d, finish_reason=%s): %r",
                            model_name,
                            key_index + 1,
                            finish_reason,
                            raw_text,
                        )
                        last_error = ValueError(
                            f"Unparseable JSON from {model_name}"
                        )
                        break

                    except errors.APIError as e:
                        code = getattr(e, "code", None)

                        if code in KEY_LEVEL_STATUS_CODES:
                            logger.warning(
                                "Gemini API key #%d rejected/exhausted "
                                "(HTTP %s): %s. Trying next key.",
                                key_index + 1,
                                code,
                                e,
                            )
                            last_error = e
                            skip_key = True
                            break  # stop trying models for this key

                        logger.warning(
                            "Gemini model '%s' unavailable (key #%d, HTTP %s): "
                            "%s. Trying next model.",
                            model_name,
                            key_index + 1,
                            code,
                            e,
                        )
                        last_error = e
                        break  # API error: don't retry same model

        raise AllKeysExhaustedError(
            f"All Gemini API keys/models failed. Last error: {last_error}"
        )

    @staticmethod
    def _extract_response_data(
        response: Any,
    ) -> tuple[Optional[dict], str, Optional[str]]:
        raw_text = response.text or ""
        finish_reason = GeminiClassifier._get_finish_reason(response)

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            data = GeminiClassifier._coerce_to_dict(parsed)
            if data is not None:
                return data, raw_text, finish_reason

        if raw_text:
            try:
                data = json.loads(raw_text)
                if isinstance(data, dict):
                    return data, raw_text, finish_reason
            except json.JSONDecodeError:
                recovered = GeminiClassifier._recover_classification_dict(raw_text)
                if recovered is not None:
                    logger.warning(
                        "Recovered classification from malformed Gemini JSON"
                    )
                    return recovered, raw_text, finish_reason

        return None, raw_text, finish_reason

    @staticmethod
    def _coerce_to_dict(parsed: Any) -> Optional[dict]:
        if isinstance(parsed, dict):
            return parsed
        model_dump = getattr(parsed, "model_dump", None)
        if callable(model_dump):
            data = model_dump()
            return data if isinstance(data, dict) else None
        return None

    @staticmethod
    def _get_finish_reason(response: Any) -> Optional[str]:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is None:
            return None
        return str(finish_reason).split(".")[-1]

    @staticmethod
    def _was_truncated(finish_reason: Optional[str]) -> bool:
        return finish_reason == "MAX_TOKENS"

    @staticmethod
    def _classification_from_data(data: dict) -> ClassificationResult:
        paper_size = data.get("paper_size") or None
        if paper_size not in (None, "Short", "Long"):
            paper_size = None
        return ClassificationResult(
            is_print_request=bool(data.get("is_print_request", False)),
            paper_size=paper_size,
            reason=str(data.get("reason", "")),
        )

    @staticmethod
    def _reply_from_data(data: dict) -> ReplyDecision:
        copies = data.get("copies")
        decision = str(data.get("decision", "unclear"))
        if decision not in ("approve", "cancel", "unclear"):
            decision = "unclear"
        fit_on_short = data.get("fit_on_short")
        if fit_on_short is not None:
            fit_on_short = bool(fit_on_short)
        return ReplyDecision(
            decision=decision,
            copies=int(copies) if copies is not None else None,
            fit_on_short=fit_on_short,
        )

    @staticmethod
    def _recover_classification_dict(raw_text: str) -> Optional[dict]:
        if not re.search(r'"is_print_request"\s*:\s*true\b', raw_text, re.IGNORECASE):
            return None

        paper_size = None
        paper_match = re.search(
            r'"paper_size"\s*:\s*"(Short|Long)"', raw_text, re.IGNORECASE
        )
        if paper_match:
            paper_size = paper_match.group(1).capitalize()

        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', raw_text)
        reason = (
            reason_match.group(1)
            if reason_match
            else "Recovered from partial model response"
        )

        return {
            "is_print_request": True,
            "paper_size": paper_size,
            "reason": reason,
        }
