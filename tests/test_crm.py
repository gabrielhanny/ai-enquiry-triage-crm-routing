from datetime import datetime, timezone

from app.crm import MockCRM
from app.models import Channel, EnquiryCategory, NormalizedEnquiry, TriageResult


def make_enquiry(enquiry_id: str, email: str = "a@b.com") -> NormalizedEnquiry:
    return NormalizedEnquiry(
        enquiry_id=enquiry_id,
        channel=Channel.EMAIL,
        sender_name="Test",
        sender_email=email,
        text="hello",
        received_at=datetime.now(timezone.utc),
    )


def test_first_sales_enquiry_creates_a_lead():
    crm = MockCRM()
    triage = TriageResult(category=EnquiryCategory.SALES, confidence=0.9, company_name="Acme", reasoning="x")
    action = crm.process(make_enquiry("e1"), triage)
    assert action.action == "create_lead"
    assert not action.is_duplicate
    assert action.record_id.startswith("LEAD-")


def test_repeat_sales_enquiry_from_same_email_is_flagged_duplicate():
    crm = MockCRM()
    triage = TriageResult(category=EnquiryCategory.SALES, confidence=0.9, company_name="Acme", reasoning="x")
    first = crm.process(make_enquiry("e1"), triage)
    second = crm.process(make_enquiry("e2"), triage)
    assert second.is_duplicate
    assert second.action == "attach_to_existing"
    assert second.record_id == first.record_id


def test_different_categories_from_same_email_create_separate_records():
    crm = MockCRM()
    sales = TriageResult(category=EnquiryCategory.SALES, confidence=0.9, company_name="Acme", reasoning="x")
    support = TriageResult(category=EnquiryCategory.SUPPORT, confidence=0.9, issue_summary="bug", reasoning="x")
    lead = crm.process(make_enquiry("e1"), sales)
    ticket = crm.process(make_enquiry("e2"), support)
    assert lead.record_id != ticket.record_id
    assert not ticket.is_duplicate


def test_missing_email_falls_back_to_unknown_bucket():
    crm = MockCRM()
    triage = TriageResult(category=EnquiryCategory.SUPPORT, confidence=0.9, issue_summary="bug", reasoning="x")
    action = crm.process(make_enquiry("e1", email=None), triage)
    assert action.action == "create_ticket"
