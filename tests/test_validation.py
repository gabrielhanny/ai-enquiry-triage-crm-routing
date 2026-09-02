from datetime import datetime, timezone

from app.models import Channel, EnquiryCategory, NormalizedEnquiry, TriageResult
from app.validation import (
    CONFIDENCE_THRESHOLD,
    apply_category_safeguards,
    decide_route,
    is_contact_detail_correction,
    validate_triage,
)


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


# --- Category safeguard: contact-detail corrections must not be "infrastructure" ---


def test_contact_correction_is_detected_generically():
    assert is_contact_detail_correction(
        "Just correcting my number from the web form. It is 0411 999 102, "
        "not 0411 999 120. Please use this email address going forward."
    )
    assert is_contact_detail_correction("I'd like to update my email address on file.")


def test_unrelated_internal_system_report_is_not_a_contact_correction():
    assert not is_contact_detail_correction(
        "HubSpot sync job failed at 02:14. Error: OAuth token expired. "
        "146 records remain unsynchronised. Retry disabled after three failures."
    )


def test_infrastructure_contact_correction_is_reclassified_to_operations():
    enquiry = make_enquiry().model_copy(
        update={
            "text": "Just correcting my number from the web form. It is 0411 999 102, "
            "not 0411 999 120. Please use this email address going forward."
        }
    )
    triage = TriageResult(
        category=EnquiryCategory.INFRASTRUCTURE, confidence=0.9, phone="0411 999 102", reasoning="x"
    )
    corrected = apply_category_safeguards(enquiry, triage)
    assert corrected.category == EnquiryCategory.OPERATIONS
    # Nothing else about the triage result should change.
    assert corrected.phone == "0411 999 102"
    assert corrected.confidence == 0.9


def test_genuine_infrastructure_report_is_not_reclassified():
    enquiry = make_enquiry().model_copy(
        update={"text": "HubSpot sync job failed at 02:14. Error: OAuth token expired."}
    )
    triage = TriageResult(category=EnquiryCategory.INFRASTRUCTURE, confidence=0.9, reasoning="x")
    corrected = apply_category_safeguards(enquiry, triage)
    assert corrected.category == EnquiryCategory.INFRASTRUCTURE


def test_contact_correction_phrasing_only_matters_for_infrastructure_category():
    # The safeguard only overrides `infrastructure` — it must not touch any
    # other category, even if the text happens to mention a correction.
    enquiry = make_enquiry().model_copy(update={"text": "Please correct my phone number to 0400 000 000."})
    triage = TriageResult(category=EnquiryCategory.SALES, confidence=0.9, company_name="Acme", reasoning="x")
    corrected = apply_category_safeguards(enquiry, triage)
    assert corrected.category == EnquiryCategory.SALES
