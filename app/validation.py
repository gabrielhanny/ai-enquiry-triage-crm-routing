"""Deterministic validation and routing rules.

The LLM may *suggest* a category, extracted fields, and missing information,
but this module is the sole source of truth for whether an enquiry is valid,
complete, and how it should be routed. The LLM's own `missing_information`
claim is never trusted directly — required fields are re-derived here.
"""
from __future__ import annotations

import re

from .models import EnquiryCategory, NormalizedEnquiry, TriageResult, ValidationResult

CONFIDENCE_THRESHOLD = 0.6

# Generic phrasing for a customer correcting their own contact details
# (phone/email/address) — deliberately not tied to any specific sender,
# company, or wording so it catches similar corrections, not just one case.
_CONTACT_CORRECTION_RE = re.compile(
    r"\b(correct(?:ing|ed)?|updat(?:e|ing|ed)|chang(?:e|ing|ed))\b[^.]{0,40}"
    r"\b(number|phone|email|address|contact|details?)\b"
    r"|\bplease use (?:this|the) email\b"
    r"|\b(?:number|phone)\b[^.]{0,20}\bnot\b[^.]{0,20}\d{2,}",
    re.IGNORECASE,
)


def is_contact_detail_correction(text: str) -> bool:
    """Deterministic signal that an enquiry is a customer correcting their
    own contact details, as opposed to reporting a problem with the
    business's own internal systems."""
    return bool(_CONTACT_CORRECTION_RE.search(text))


def apply_category_safeguards(enquiry: NormalizedEnquiry, triage: TriageResult) -> TriageResult:
    """Deterministic override applied after LLM triage, before validation
    and routing. `infrastructure` is reserved for reports about the
    business's OWN internal systems/tools — a customer correcting their own
    contact details must never land there, regardless of what the LLM
    guessed. Reclassified to `operations` (administrative/contact-record
    housekeeping), which routes to the operations owner via the existing
    staff-directory keyword matching. Everything else about the triage
    result — extracted fields, confidence, reasoning — is left untouched."""
    if triage.category == EnquiryCategory.INFRASTRUCTURE and is_contact_detail_correction(enquiry.text):
        return triage.model_copy(update={"category": EnquiryCategory.OPERATIONS})
    return triage


def validate_triage(enquiry: NormalizedEnquiry, triage: TriageResult) -> ValidationResult:
    reasons: list[str] = []
    missing: list[str] = []

    if triage.confidence < CONFIDENCE_THRESHOLD:
        reasons.append(f"confidence {triage.confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}")

    if triage.category == EnquiryCategory.SALES:
        if not enquiry.sender_email:
            missing.append("sender_email")
        if not (triage.company_name or triage.product_interest):
            missing.append("company_name_or_product_interest")
    elif triage.category == EnquiryCategory.SUPPORT:
        if not enquiry.sender_email:
            missing.append("sender_email")
        if not triage.issue_summary:
            missing.append("issue_summary")
    elif triage.category == EnquiryCategory.TECHNICAL:
        if not triage.issue_summary:
            missing.append("issue_summary")
    # operations, infrastructure, other, and junk have no hard requirements
    # beyond the confidence threshold — they are typically internal,
    # low-risk, or too varied to force a fixed schema on.

    is_complete = len(missing) == 0
    is_valid = is_complete and triage.confidence >= CONFIDENCE_THRESHOLD

    if missing:
        reasons.append(f"missing required fields: {', '.join(missing)}")

    return ValidationResult(is_valid=is_valid, is_complete=is_complete, missing_fields=missing, reasons=reasons)


def decide_route(triage: TriageResult, validation: ValidationResult) -> str:
    """Returns "archive", "clarify", or the category name (e.g. "sales")."""
    if triage.category == EnquiryCategory.JUNK:
        return "archive"
    if not validation.is_valid:
        return "clarify"
    return triage.category.value
