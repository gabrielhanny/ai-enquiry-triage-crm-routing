from app.crm_match import Candidate, detect_conflicts, match_enquiry, normalize_company, normalize_phone
from app.models import Channel, EnquiryCategory, NormalizedEnquiry, TriageResult
from datetime import datetime, timezone


def make_enquiry(email: str | None) -> NormalizedEnquiry:
    return NormalizedEnquiry(
        enquiry_id="e1",
        channel=Channel.EMAIL,
        sender_name="Test",
        sender_email=email,
        text="hello",
        received_at=datetime.now(timezone.utc),
    )


def test_normalize_company_strips_legal_suffix_and_case():
    assert normalize_company("Hume Logistics Pty Ltd") == "hume logistics"
    assert normalize_company("Hume Logistic") == "hume logistic"


def test_normalize_phone_strips_non_digits():
    assert normalize_phone("0400 111 020") == "0400111020"


def test_exact_email_match_is_likely():
    enquiry = make_enquiry("amelia.grant@humelogistics.example")
    triage = TriageResult(category=EnquiryCategory.SALES, confidence=0.9, reasoning="x")
    candidate = Candidate("C001", company="Hume Logistics Pty Ltd", email="amelia.grant@humelogistics.example", phone="0400 111 020")
    result = match_enquiry(enquiry, triage, [candidate])
    assert result.status == "likely_match"
    assert result.best.record_id == "C001"


def test_no_signals_at_all_is_no_match():
    enquiry = make_enquiry("nobody@nowhere.example")
    triage = TriageResult(category=EnquiryCategory.SALES, confidence=0.9, reasoning="x")
    candidate = Candidate("C001", company="Acme Corp", email="someone@acme.example", phone="0400 000 000")
    result = match_enquiry(enquiry, triage, [candidate])
    assert result.status == "no_match"
    assert result.best is None


def test_weak_domain_only_signal_is_possible_not_likely():
    # Different sender within the same email domain, no other overlap:
    # plausible same organisation, but should not be forced as certain.
    enquiry = make_enquiry("sam@harbourcoldstores.example")
    triage = TriageResult(category=EnquiryCategory.OTHER, confidence=0.8, reasoning="x")
    candidate = Candidate("LEAD-0001", email="facilities@harbourcoldstores.example", phone="0411 999 120")
    result = match_enquiry(enquiry, triage, [candidate])
    assert result.status == "possible_match"


def test_does_not_force_a_match_when_evidence_is_too_thin():
    enquiry = make_enquiry("info@smallcafe.example")
    triage = TriageResult(category=EnquiryCategory.SALES, confidence=0.8, reasoning="x")
    candidate = Candidate("C003", company="Greenfields Foods Pty Ltd", email="rohan@greenfieldsfoods.example")
    result = match_enquiry(enquiry, triage, [candidate])
    assert result.status == "no_match"


def test_close_scores_produce_an_ambiguity_note():
    enquiry = make_enquiry("a.grant@humelogistics.example")
    triage = TriageResult(category=EnquiryCategory.SALES, confidence=0.9, company_name="Hume Logistic", reasoning="x")
    c001 = Candidate("C001", company="Hume Logistics Pty Ltd", email="amelia.grant@humelogistics.example", phone="0400 111 020")
    c002 = Candidate("C002", company="Hume Logistic", email="a.grant@humelogistics.example")
    result = match_enquiry(enquiry, triage, [c001, c002])
    assert result.status == "likely_match"
    assert len(result.candidates) == 2
    assert result.notes, "expected an ambiguity note when two candidates score closely"


def test_detect_conflicts_flags_differing_phone_but_not_new_information():
    enquiry = make_enquiry("sam@harbourcoldstores.example")
    triage = TriageResult(category=EnquiryCategory.OTHER, confidence=0.8, phone="0411 999 102", reasoning="x")
    candidate = Candidate("LEAD-0001", email="sam@harbourcoldstores.example", phone="0411 999 120")
    conflicts = detect_conflicts(enquiry, triage, candidate, source_id="E010")
    fields = {c.field for c in conflicts}
    assert "phone" in fields
    phone_conflict = next(c for c in conflicts if c.field == "phone")
    assert phone_conflict.existing_value == "0411 999 120"
    assert phone_conflict.new_value == "0411 999 102"
    assert phone_conflict.source == "E010"


def test_detect_conflicts_ignores_blank_existing_value():
    # CRM has no phone on file at all — a supplied phone is new information,
    # not a conflict, so it must not be flagged.
    enquiry = make_enquiry("a.grant@humelogistics.example")
    triage = TriageResult(category=EnquiryCategory.SALES, confidence=0.9, phone="0400 111 020", reasoning="x")
    candidate = Candidate("C002", email="a.grant@humelogistics.example", phone="")
    conflicts = detect_conflicts(enquiry, triage, candidate, source_id="E002")
    assert conflicts == []
