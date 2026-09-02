"""Pydantic data models shared across the enquiry triage workflow."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


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
    TECHNICAL = "technical"
    OPERATIONS = "operations"
    INFRASTRUCTURE = "infrastructure"
    OTHER = "other"
    JUNK = "junk"


class TriageResult(BaseModel):
    """LLM output. Advisory only — never treated as a source of truth."""

    category: EnquiryCategory
    confidence: float = Field(ge=0.0, le=1.0)
    company_name: Optional[str] = None
    contact_name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
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


# Placeholder record_id for a CRM action that has been decided/recommended
# but not yet executed — the real id (if any) is only assigned once the
# mutation actually happens, after human approval.
PENDING_CRM_RECORD_ID = "PENDING-APPROVAL"


class CRMAction(BaseModel):
    """Result of a mock CRM operation."""

    action: str  # create_lead | create_ticket | attach_to_existing_crm_record | flag_for_review | not_applied
    record_id: str
    is_duplicate: bool
    details: dict = Field(default_factory=dict)


class ApprovalRecord(BaseModel):
    """Result of the human-in-the-loop approval gate."""

    approved: bool
    approver: str
    note: Optional[str] = None


# --- Test 2: ingested reference data (untrusted, loaded verbatim from fixtures) ---


class StaffMember(BaseModel):
    """One row from staff_directory.json."""

    name: str
    role: str
    owns: str


class CRMRecord(BaseModel):
    """One row from crm.csv — a pre-existing (fictional) CRM record, treated
    as untrusted, possibly messy reference data, not as ground truth."""

    id: str
    company: str
    contact: str
    email: str
    phone: str
    location: str
    status: str
    service: str
    state: str


class EmailMessage(BaseModel):
    """One row from emails.json, as supplied — unnormalized."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    from_: str = Field(alias="from")
    subject: Optional[str] = None
    body: str
    attachment: Optional[str] = None


# --- Test 2: CRM matching and conflict detection (deterministic, not LLM-controlled) ---


class MatchCandidate(BaseModel):
    """One scored candidate CRM record for an enquiry."""

    record_id: str
    company: str
    score: float
    signals: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    """Deterministic CRM match verdict for an enquiry. Never forces an
    uncertain match — status reflects the strength of the evidence."""

    status: str  # no_match | possible_match | likely_match
    candidates: list[MatchCandidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def best(self) -> Optional[MatchCandidate]:
        return self.candidates[0] if self.candidates else None


class Conflict(BaseModel):
    """A factual disagreement between new evidence and an existing CRM
    record. Both values are preserved — neither is silently overwritten."""

    field: str
    existing_value: str
    new_value: str
    source: str
    note: Optional[str] = None


class OwnerRecommendation(BaseModel):
    """Deterministic staff-routing suggestion derived from staff_directory.json."""

    owner: Optional[str] = None
    confidence: float = 0.0
    reasoning: str
