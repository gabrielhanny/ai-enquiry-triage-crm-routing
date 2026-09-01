"""Pydantic data models shared across the enquiry triage workflow."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Channel(str, Enum):
    EMAIL = "email"
    WEB_FORM = "web_form"
    MESSAGING = "messaging"


class RawEnquiry(BaseModel):
    """Unnormalized input as received from a source channel."""

    channel: Channel
    sender_name: Optional[str] = None
    sender_email: Optional[str] = None
    subject: Optional[str] = None
    body: str
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NormalizedEnquiry(BaseModel):
    """Common shape used internally regardless of source channel."""

    enquiry_id: str
    channel: Channel
    sender_name: Optional[str] = None
    sender_email: Optional[str] = None
    text: str
    received_at: datetime


class EnquiryCategory(str, Enum):
    SALES = "sales"
    SUPPORT = "support"
    JUNK = "junk"


class TriageResult(BaseModel):
    """LLM output. Advisory only — never treated as a source of truth."""

    category: EnquiryCategory
    confidence: float = Field(ge=0.0, le=1.0)
    company_name: Optional[str] = None
    product_interest: Optional[str] = None
    issue_summary: Optional[str] = None
    missing_information: list[str] = Field(default_factory=list)
    reasoning: str


class ValidationResult(BaseModel):
    """Deterministic verdict computed by application code, not the LLM."""

    is_valid: bool
    is_complete: bool
    missing_fields: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class CRMAction(BaseModel):
    """Result of a mock CRM operation."""

    action: str  # create_lead | create_ticket | attach_to_existing
    record_id: str
    is_duplicate: bool
    details: dict = Field(default_factory=dict)


class ApprovalRecord(BaseModel):
    """Result of the human-in-the-loop approval gate."""

    approved: bool
    approver: str
    note: Optional[str] = None
