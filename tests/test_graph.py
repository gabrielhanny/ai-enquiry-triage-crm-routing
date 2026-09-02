"""Exercises the compiled LangGraph workflow end-to-end with a fake LLM.

The real OpenAI client is never invoked in tests — cost- and
latency-sensitive integration is stubbed at the app.llm module boundary.
"""
from app import llm
from app.graph import build_graph
from app.models import Channel, CRMRecord, EnquiryCategory, RawEnquiry, StaffMember, TriageResult


def make_raw(**overrides) -> RawEnquiry:
    defaults = dict(
        channel=Channel.EMAIL,
        sender_name="Test",
        sender_email="test@example.com",
        subject="hi",
        body="hello",
    )
    defaults.update(overrides)
    return RawEnquiry(**defaults)


def test_complete_sales_enquiry_routes_to_sales_and_approves(monkeypatch):
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(
            category=EnquiryCategory.SALES, confidence=0.9, company_name="Acme", reasoning="clear intent"
        ),
    )
    monkeypatch.setattr(llm, "draft_response", lambda enquiry, triage, crm_action: "draft response")

    workflow = build_graph()
    result = workflow.invoke({"raw": make_raw()})

    assert result["route"] == "sales"
    assert result["crm_action"].action == "create_lead"
    assert result["approval"].approved is True


def test_incomplete_enquiry_routes_to_clarify(monkeypatch):
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(category=EnquiryCategory.SALES, confidence=0.9, reasoning="vague"),
    )
    monkeypatch.setattr(llm, "draft_clarification", lambda enquiry, missing: "please clarify")

    workflow = build_graph()
    result = workflow.invoke({"raw": make_raw(sender_email=None, body="pricing?")})

    assert result["route"] == "clarify"
    assert "crm_action" not in result


def test_junk_enquiry_routes_to_archive(monkeypatch):
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(category=EnquiryCategory.JUNK, confidence=0.95, reasoning="spam"),
    )

    workflow = build_graph()
    result = workflow.invoke({"raw": make_raw(body="CLICK HERE to win a prize!!!")})

    assert result["route"] == "archive"
    assert "approval" not in result


def test_llm_failure_fails_safe_to_clarify(monkeypatch):
    def boom(enquiry):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(llm, "triage_enquiry", boom)
    monkeypatch.setattr(llm, "draft_clarification", lambda enquiry, missing: "please clarify (fallback)")

    workflow = build_graph()
    result = workflow.invoke({"raw": make_raw()})

    assert result["route"] == "clarify"
    assert result["error"] == "LLM unavailable"
    assert result["approval"].approved is True


def test_repeat_sales_enquiry_is_flagged_duplicate_in_graph(monkeypatch):
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(
            category=EnquiryCategory.SALES, confidence=0.9, company_name="Acme", reasoning="clear intent"
        ),
    )
    monkeypatch.setattr(llm, "draft_response", lambda enquiry, triage, crm_action: "draft response")

    workflow = build_graph()
    raw = make_raw()
    first = workflow.invoke({"raw": raw})
    second = workflow.invoke({"raw": raw})

    assert not first["crm_action"].is_duplicate
    assert second["crm_action"].is_duplicate
    assert second["crm_action"].record_id == first["crm_action"].record_id


# --- Test 2 additions: expanded categories, CRM matching, conflicts, owner routing ---


def make_crm_record(**overrides) -> CRMRecord:
    defaults = dict(
        id="C001",
        company="Hume Logistics Pty Ltd",
        contact="Amelia Grant",
        email="amelia.grant@humelogistics.example",
        phone="0400 111 020",
        location="Melbourne VIC",
        status="Prospect",
        service="Commercial Solar",
        state="Open",
    )
    defaults.update(overrides)
    return CRMRecord(**defaults)


def make_staff() -> list[StaffMember]:
    return [
        StaffMember(name="Matt Cooper", role="Founder", owns="major commercial opportunities and strategic partnerships"),
        StaffMember(name="Ties Rahardjo", role="Executive Operations Coordinator", owns="scheduling, administration, logistics and general operational enquiries"),
        StaffMember(name="Ali Pratama", role="Senior Business Analyst", owns="CRM, systems, data, workflows and infrastructure issues"),
    ]


def test_technical_enquiry_with_issue_summary_routes_to_technical(monkeypatch):
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(
            category=EnquiryCategory.TECHNICAL,
            confidence=0.9,
            issue_summary="Confirm THD limits at point of common coupling",
            reasoning="engineering question",
        ),
    )
    monkeypatch.setattr(llm, "draft_response", lambda enquiry, triage, crm_action: "draft response")

    workflow = build_graph()
    result = workflow.invoke({"raw": make_raw(sender_email=None, body="harmonics question")})

    assert result["route"] == "technical"
    assert result["approval"].approved is True


def test_technical_enquiry_without_issue_summary_routes_to_clarify(monkeypatch):
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(category=EnquiryCategory.TECHNICAL, confidence=0.9, reasoning="vague technical ask"),
    )
    monkeypatch.setattr(llm, "draft_clarification", lambda enquiry, missing: "please clarify")

    workflow = build_graph()
    result = workflow.invoke({"raw": make_raw(body="can you help?")})

    assert result["route"] == "clarify"
    assert "issue_summary" in result["validation"].missing_fields


def test_infrastructure_enquiry_has_no_required_fields(monkeypatch):
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(
            category=EnquiryCategory.INFRASTRUCTURE, confidence=0.9, reasoning="internal system alert"
        ),
    )
    monkeypatch.setattr(llm, "draft_response", lambda enquiry, triage, crm_action: "draft response")

    workflow = build_graph()
    result = workflow.invoke({"raw": make_raw(sender_email="alerts@internal.example", body="sync failed")})

    assert result["route"] == "infrastructure"
    assert result["validation"].is_valid is True


def test_likely_crm_match_attaches_to_existing_record_and_skips_new_creation(monkeypatch):
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(
            category=EnquiryCategory.SALES, confidence=0.9, product_interest="solar", reasoning="clear intent"
        ),
    )
    monkeypatch.setattr(llm, "draft_response", lambda enquiry, triage, crm_action: "draft response")

    workflow = build_graph()
    crm_reference = [make_crm_record()]
    result = workflow.invoke(
        {
            "raw": make_raw(sender_email="amelia.grant@humelogistics.example"),
            "crm_reference": crm_reference,
        }
    )

    assert result["match_result"].status == "likely_match"
    assert result["crm_action"].action == "attach_to_existing_crm_record"
    assert result["crm_action"].record_id == "C001"
    assert result["crm_action"].is_duplicate is True


def test_weak_signal_crm_match_is_flagged_for_review_not_forced(monkeypatch):
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(category=EnquiryCategory.OTHER, confidence=0.8, reasoning="ambiguous"),
    )
    monkeypatch.setattr(llm, "draft_response", lambda enquiry, triage, crm_action: "draft response")

    workflow = build_graph()
    # Same email domain as the CRM record but a different mailbox and no
    # other overlapping signal — should not be forced into a confident match.
    crm_reference = [make_crm_record(email="amelia.grant@humelogistics.example", phone="0400 111 020")]
    result = workflow.invoke(
        {
            "raw": make_raw(sender_email="someoneelse@humelogistics.example"),
            "crm_reference": crm_reference,
        }
    )

    assert result["match_result"].status == "possible_match"
    assert result["crm_action"].action == "flag_for_review"
    assert result["crm_action"].is_duplicate is False


def test_conflicting_phone_number_is_preserved_not_overwritten(monkeypatch):
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(
            category=EnquiryCategory.OTHER, confidence=0.9, phone="0411 999 102", reasoning="correcting phone number"
        ),
    )
    monkeypatch.setattr(llm, "draft_response", lambda enquiry, triage, crm_action: "draft response")

    workflow = build_graph()
    crm_reference = [
        make_crm_record(
            id="C009", company="", contact="Sam", email="sam@harbourcoldstores.example", phone="0411 999 120", location=""
        )
    ]
    result = workflow.invoke(
        {
            "raw": make_raw(sender_email="sam@harbourcoldstores.example"),
            "crm_reference": crm_reference,
        }
    )

    conflicts = result["conflicts"]
    assert any(c.field == "phone" and c.existing_value == "0411 999 120" and c.new_value == "0411 999 102" for c in conflicts)
    # The old value must still be visible — it was never silently replaced.
    assert result["crm_action"].record_id == "C009"


def test_owner_is_recommended_from_staff_directory_when_supplied(monkeypatch):
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(category=EnquiryCategory.SALES, confidence=0.9, company_name="Acme", reasoning="x"),
    )
    monkeypatch.setattr(llm, "draft_response", lambda enquiry, triage, crm_action: "draft response")

    workflow = build_graph()
    result = workflow.invoke({"raw": make_raw(), "staff_directory": make_staff()})

    assert result["owner"].owner == "Matt Cooper"


def test_no_staff_directory_supplied_leaves_owner_unset(monkeypatch):
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(category=EnquiryCategory.SALES, confidence=0.9, company_name="Acme", reasoning="x"),
    )
    monkeypatch.setattr(llm, "draft_response", lambda enquiry, triage, crm_action: "draft response")

    workflow = build_graph()
    result = workflow.invoke({"raw": make_raw()})

    assert "owner" not in result


def test_junk_enquiry_never_reaches_approval_or_crm_even_with_reference_data(monkeypatch):
    """Permission/approval boundary: junk must archive without touching CRM
    or approval, regardless of what reference data is supplied."""
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(category=EnquiryCategory.JUNK, confidence=0.95, reasoning="spam"),
    )

    workflow = build_graph()
    result = workflow.invoke(
        {
            "raw": make_raw(body="buy leads now"),
            "crm_reference": [make_crm_record()],
            "staff_directory": make_staff(),
        }
    )

    assert result["route"] == "archive"
    assert "approval" not in result
    assert "crm_action" not in result


def test_audit_trail_captures_match_conflicts_and_owner(monkeypatch):
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(
            category=EnquiryCategory.SALES, confidence=0.9, product_interest="solar", reasoning="x"
        ),
    )
    monkeypatch.setattr(llm, "draft_response", lambda enquiry, triage, crm_action: "draft response")

    workflow = build_graph()
    workflow.invoke(
        {
            "raw": make_raw(sender_email="amelia.grant@humelogistics.example"),
            "crm_reference": [make_crm_record()],
            "staff_directory": make_staff(),
            "source_email_id": "E001",
        }
    )

    from app.audit import fetch_all_events
    import json

    events = fetch_all_events()
    assert len(events) == 1
    details = json.loads(events[0]["details"])
    assert details["source_email_id"] == "E001"
    assert details["match_status"] == "likely_match"
    assert details["owner"] == "Matt Cooper"


# --- Architectural fix: approval must precede consequential CRM mutation ---


def test_no_crm_mutation_occurs_before_approval(monkeypatch):
    """The approval gate is called before any CRM record exists — proving
    analyze_crm (which runs first) is genuinely read-only."""
    from app import approval as approval_mod
    from app import crm as crm_mod
    from app.models import ApprovalRecord

    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(
            category=EnquiryCategory.SALES, confidence=0.9, product_interest="solar", reasoning="x"
        ),
    )
    monkeypatch.setattr(llm, "draft_response", lambda enquiry, triage, crm_action: "draft response")

    def spy_approval(*, summary, approver="demo-reviewer"):
        assert crm_mod.crm.all_records() == [], "CRM was mutated before approval was requested"
        return ApprovalRecord(approved=True, approver=approver, note="test-approved")

    monkeypatch.setattr(approval_mod, "request_human_approval", spy_approval)

    workflow = build_graph()
    result = workflow.invoke({"raw": make_raw()})

    assert result["crm_action"].action == "create_lead"
    assert len(crm_mod.crm.all_records()) == 1


def test_rejected_approval_prevents_new_record_creation(monkeypatch):
    from app import approval as approval_mod
    from app import crm as crm_mod
    from app.models import ApprovalRecord

    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(
            category=EnquiryCategory.SALES, confidence=0.9, product_interest="solar", reasoning="x"
        ),
    )
    monkeypatch.setattr(llm, "draft_response", lambda enquiry, triage, crm_action: "draft response")
    monkeypatch.setattr(
        approval_mod,
        "request_human_approval",
        lambda *, summary, approver="demo-reviewer": ApprovalRecord(approved=False, approver=approver, note="rejected"),
    )

    workflow = build_graph()
    result = workflow.invoke({"raw": make_raw()})

    assert result["approval"].approved is False
    assert result["crm_action"].action == "not_applied"
    assert crm_mod.crm.all_records() == []


def test_rejected_approval_prevents_attach_to_existing_record(monkeypatch):
    """A second enquiry that would attach to a record created by the first
    must not actually be linked to it if approval is withheld."""
    from app import approval as approval_mod
    from app import crm as crm_mod
    from app.models import ApprovalRecord

    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(
            category=EnquiryCategory.SALES, confidence=0.9, product_interest="solar", reasoning="x"
        ),
    )
    monkeypatch.setattr(llm, "draft_response", lambda enquiry, triage, crm_action: "draft response")

    workflow = build_graph()
    raw = make_raw()

    first = workflow.invoke({"raw": raw})
    assert first["crm_action"].action == "create_lead"
    assert crm_mod.crm.all_records()[0]["enquiry_ids"] == [first["enquiry"].enquiry_id]

    monkeypatch.setattr(
        approval_mod,
        "request_human_approval",
        lambda *, summary, approver="demo-reviewer": ApprovalRecord(approved=False, approver=approver, note="rejected"),
    )
    second = workflow.invoke({"raw": raw})

    assert second["match_result"].status == "likely_match"
    assert second["crm_action"].action == "not_applied"
    stored = crm_mod.crm.all_records()[0]
    assert second["enquiry"].enquiry_id not in stored["enquiry_ids"]


def test_clarify_path_never_touches_apply_crm_action(monkeypatch):
    """apply_crm_action must no-op cleanly for the clarify path, which never
    computes a recommended_action."""
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(category=EnquiryCategory.SALES, confidence=0.9, reasoning="vague"),
    )
    monkeypatch.setattr(llm, "draft_clarification", lambda enquiry, missing: "please clarify")

    workflow = build_graph()
    result = workflow.invoke({"raw": make_raw(sender_email=None, body="pricing?")})

    assert result["route"] == "clarify"
    assert "recommended_action" not in result
    assert "crm_action" not in result


# --- Architectural fix: contact-detail corrections must not be "infrastructure" ---


def test_contact_correction_is_not_routed_to_infrastructure_even_if_llm_mislabels_it(monkeypatch):
    monkeypatch.setattr(
        llm,
        "triage_enquiry",
        lambda enquiry: TriageResult(
            category=EnquiryCategory.INFRASTRUCTURE,
            confidence=0.9,
            phone="0411 999 102",
            reasoning="(simulating an LLM misclassification, as seen on E010)",
        ),
    )
    monkeypatch.setattr(llm, "draft_response", lambda enquiry, triage, crm_action: "draft response")

    workflow = build_graph()
    result = workflow.invoke(
        {
            "raw": make_raw(
                body="Just correcting my number from the web form. It is 0411 999 102, "
                "not 0411 999 120. Please use this email address going forward."
            ),
            "staff_directory": make_staff(),
        }
    )

    assert result["route"] != "infrastructure"
    assert result["route"] == "operations"
    assert result["triage"].category == EnquiryCategory.OPERATIONS
    assert result["owner"].owner == "Ties Rahardjo"
