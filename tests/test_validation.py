from datetime import datetime, timezone

from app.models import Channel, EnquiryCategory, NormalizedEnquiry, TriageResult
from app.validation import CONFIDENCE_THRESHOLD, decide_route, validate_triage


def make_enquiry(sender_email="a@b.com") -> NormalizedEnquiry:
    return NormalizedEnquiry(
        enquiry_id="e1",
        channel=Channel.EMAIL,
        sender_name="Test",
        sender_email=sender_email,
        text="hello",
        received_at=datetime.now(timezone.utc),
    )


def test_complete_sales_is_valid_and_routes_to_sales():
    enquiry = make_enquiry()
    triage = TriageResult(category=EnquiryCategory.SALES, confidence=0.9, company_name="Acme", reasoning="x")
    result = validate_triage(enquiry, triage)
    assert result.is_valid
    assert result.missing_fields == []
    assert decide_route(triage, result) == "sales"


def test_sales_missing_email_is_incomplete_and_routes_to_clarify():
    enquiry = make_enquiry(sender_email=None)
    triage = TriageResult(category=EnquiryCategory.SALES, confidence=0.9, company_name="Acme", reasoning="x")
    result = validate_triage(enquiry, triage)
    assert not result.is_valid
    assert "sender_email" in result.missing_fields
    assert decide_route(triage, result) == "clarify"


def test_low_confidence_forces_clarify_even_if_fields_present():
    enquiry = make_enquiry()
    triage = TriageResult(
        category=EnquiryCategory.SUPPORT,
        confidence=CONFIDENCE_THRESHOLD - 0.1,
        issue_summary="broken widget",
        reasoning="x",
    )
    result = validate_triage(enquiry, triage)
    assert not result.is_valid
    assert decide_route(triage, result) == "clarify"


def test_junk_always_routes_to_archive_regardless_of_fields():
    enquiry = make_enquiry(sender_email=None)
    triage = TriageResult(category=EnquiryCategory.JUNK, confidence=0.95, reasoning="spam")
    result = validate_triage(enquiry, triage)
    assert decide_route(triage, result) == "archive"


def test_llm_missing_information_claim_is_not_authoritative():
    # LLM claims nothing is missing, but the app must still catch a missing
    # required field on its own.
    enquiry = make_enquiry(sender_email=None)
    triage = TriageResult(
        category=EnquiryCategory.SUPPORT,
        confidence=0.9,
        issue_summary="broken widget",
        missing_information=[],
        reasoning="x",
    )
    result = validate_triage(enquiry, triage)
    assert "sender_email" in result.missing_fields
    assert not result.is_valid
