from app.audit import fetch_all_events, record_audit_event


def test_record_and_fetch_round_trip():
    record_audit_event(
        enquiry_id="e1",
        route="sales",
        category="sales",
        confidence=0.9,
        is_duplicate=False,
        crm_record_id="LEAD-0001",
        approved=True,
        approver="demo-reviewer",
        details={"draft": "hello"},
    )
    events = fetch_all_events()
    assert len(events) == 1
    assert events[0]["enquiry_id"] == "e1"
    assert events[0]["route"] == "sales"
    assert events[0]["approved"] == 1


def test_every_call_appends_a_new_row():
    record_audit_event(enquiry_id="e1", route="archive")
    record_audit_event(enquiry_id="e2", route="clarify")
    events = fetch_all_events()
    assert len(events) == 2
    assert [e["enquiry_id"] for e in events] == ["e1", "e2"]
