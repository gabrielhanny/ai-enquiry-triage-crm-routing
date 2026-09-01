"""Deterministic validation and routing rules.

The LLM may *suggest* a category, extracted fields, and missing information,
but this module is the sole source of truth for whether an enquiry is valid,
complete, and how it should be routed. The LLM's own `missing_information`
claim is never trusted directly — required fields are re-derived here.
"""
from __future__ import annotations

from .models import EnquiryCategory, NormalizedEnquiry, TriageResult, ValidationResult

CONFIDENCE_THRESHOLD = 0.6


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
    # JUNK has no required fields.

    is_complete = len(missing) == 0
    is_valid = is_complete and triage.confidence >= CONFIDENCE_THRESHOLD

    if missing:
        reasons.append(f"missing required fields: {', '.join(missing)}")

    return ValidationResult(is_valid=is_valid, is_complete=is_complete, missing_fields=missing, reasons=reasons)


def decide_route(triage: TriageResult, validation: ValidationResult) -> str:
    """Returns one of: archive, clarify, sales, support."""
    if triage.category == EnquiryCategory.JUNK:
        return "archive"
    if not validation.is_valid:
        return "clarify"
    if triage.category == EnquiryCategory.SALES:
        return "sales"
    return "support"
