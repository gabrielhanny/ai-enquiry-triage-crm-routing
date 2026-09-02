"""Loads the Test 2 fixture dataset and normalizes it into RawEnquiry input.

All loaders read the fixture files completely verbatim — no correction,
reinterpretation, or cleanup of the source data happens here. Every value
loaded from these files is untrusted input: it is passed straight into
Pydantic models (which only validate shape/type, not truthfulness), and any
interpretation of what it *means* happens later, in the LLM triage step and
in the deterministic validation/matching modules.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .models import Channel, CRMRecord, EmailMessage, RawEnquiry, StaffMember

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_FROM_RE = re.compile(r"^(?P<name>[^<]*?)\s*<(?P<email>[^>]+)>\s*$")


def load_staff_directory(path: Path | str | None = None) -> list[StaffMember]:
    path = Path(path) if path else DATA_DIR / "staff_directory.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [StaffMember(**item) for item in raw]


def load_crm_records(path: Path | str | None = None) -> list[CRMRecord]:
    path = Path(path) if path else DATA_DIR / "crm.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        return [CRMRecord(**row) for row in csv.DictReader(handle)]


def load_emails(path: Path | str | None = None) -> list[EmailMessage]:
    path = Path(path) if path else DATA_DIR / "emails.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EmailMessage(**item) for item in raw]


def load_document_text(filename: str, documents_dir: Path | str | None = None) -> str:
    documents_dir = Path(documents_dir) if documents_dir else DATA_DIR / "documents"
    return (documents_dir / filename).read_text(encoding="utf-8")


def parse_from_header(raw_from: str) -> tuple[str | None, str | None]:
    """Splits a messy 'from' string into (name, email). Handles the two
    shapes present in the dataset: 'Name <email>' and a bare email."""
    raw_from = raw_from.strip()
    match = _FROM_RE.match(raw_from)
    if match:
        name = match.group("name").strip().strip('"') or None
        return name, match.group("email").strip()
    if "@" in raw_from:
        return None, raw_from
    return raw_from or None, None


def email_to_raw_enquiry(email: EmailMessage, documents_dir: Path | str | None = None) -> RawEnquiry:
    """Converts one ingested email into a RawEnquiry, combining the body
    with attachment text as a single block of evidence. Attachment content
    is appended, not blended in undetectably — it stays clearly labelled so
    downstream reasoning (and a human reviewer) can see where it came from."""
    sender_name, sender_email = parse_from_header(email.from_)
    body = email.body
    if email.attachment:
        try:
            attachment_text = load_document_text(email.attachment, documents_dir)
            body = f"{body}\n\n--- Attachment: {email.attachment} ---\n{attachment_text}"
        except FileNotFoundError:
            body = f"{body}\n\n--- Attachment referenced but not found: {email.attachment} ---"
    return RawEnquiry(
        channel=Channel.EMAIL,
        sender_name=sender_name,
        sender_email=sender_email,
        subject=email.subject,
        body=body,
    )
