"""Exercises the compiled LangGraph workflow end-to-end with a fake LLM.

The real OpenAI client is never invoked in tests — cost- and
latency-sensitive integration is stubbed at the app.llm module boundary.
"""
from app import llm
from app.graph import build_graph
from app.models import Channel, EnquiryCategory, RawEnquiry, TriageResult


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
