"""OpenAI-backed triage, extraction, and drafting.

This is the only module allowed to talk to the LLM. It returns structured,
schema-validated output (via Pydantic) but the result is treated as advice —
`validation.py` independently decides what is actually required. On failure,
callers are expected to fall back to a safe, deterministic default rather
than blocking the workflow.
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from .models import CRMAction, NormalizedEnquiry, TriageResult

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_RETRIES = 2

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (check your .env file)")
        _client = OpenAI(api_key=api_key)
    return _client


TRIAGE_SYSTEM_PROMPT = """You are a triage assistant for a small business's inbound enquiries.
Classify each enquiry as exactly one of: sales, support, junk.

- sales: interest in purchasing, pricing, demos, or product information.
- support: an existing customer reporting a problem or asking for help.
- junk: spam, scams, or irrelevant content unrelated to the business.

Extract only information that is explicitly present in the text. Do not
invent or infer facts that are not stated. If a field is not present, leave
it null. List any information you think is missing to act on the enquiry in
`missing_information`. Give a confidence score between 0 and 1 reflecting how
sure you are of the category."""


def triage_enquiry(enquiry: NormalizedEnquiry) -> TriageResult:
    """Classify and extract structured fields from an enquiry. Retries on
    transient failure; raises after exhausting retries so the caller can
    apply its own fallback/failure-handling path."""
    client = get_client()
    user_content = (
        f"Channel: {enquiry.channel.value}\n"
        f"Sender: {enquiry.sender_name or 'unknown'} <{enquiry.sender_email or 'unknown'}>\n\n"
        f"{enquiry.text}"
    )

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.parse(
                model=MODEL,
                temperature=0,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format=TriageResult,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError("LLM returned no parsable structured output")
            return parsed
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we retry/fallback
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"LLM triage failed after {MAX_RETRIES + 1} attempts: {last_error}") from last_error


def draft_clarification(enquiry: NormalizedEnquiry, missing_fields: list[str]) -> str:
    client = get_client()
    prompt = (
        f"Write a short, polite email asking the sender for this missing "
        f"information: {', '.join(missing_fields)}.\n\n"
        f"Original message:\n{enquiry.text}\n\n"
        f"Keep it under 80 words and sign off as 'The Team'."
    )
    completion = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        max_tokens=200,
        messages=[
            {"role": "system", "content": "You write short, professional business emails."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content.strip()


def draft_response(enquiry: NormalizedEnquiry, triage: TriageResult, crm_action: CRMAction) -> str:
    client = get_client()
    prompt = (
        f"Write a short, professional acknowledgement email for a {triage.category.value} enquiry. "
        f"Reference number: {crm_action.record_id}.\n\n"
        f"Original message:\n{enquiry.text}\n\n"
        f"Keep it under 100 words and sign off as 'The Team'."
    )
    completion = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        max_tokens=220,
        messages=[
            {"role": "system", "content": "You write short, professional business emails."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content.strip()
