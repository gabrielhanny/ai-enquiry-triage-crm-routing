"""LangGraph state and workflow wiring.

    Normalize -> Triage (LLM) -> Validate (deterministic)
        -> archive                                   [junk]
        -> clarify -> human_approval                  [incomplete / low confidence / LLM failure]
        -> duplicate_check -> draft_response -> human_approval   [sales / support]
    -> audit (always, exactly once per enquiry)
"""
from __future__ import annotations

import uuid
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from . import approval as approval_mod
from . import audit
from . import crm as crm_mod
from . import llm
from . import validation as validation_mod
from .models import ApprovalRecord, CRMAction, NormalizedEnquiry, RawEnquiry, TriageResult, ValidationResult


class EnquiryState(TypedDict, total=False):
    raw: RawEnquiry
    enquiry: NormalizedEnquiry
    triage: TriageResult
    validation: ValidationResult
    route: str
    crm_action: CRMAction
    draft: str
    approval: ApprovalRecord
    error: Optional[str]


def normalize_node(state: EnquiryState) -> dict:
    raw = state["raw"]
    text = f"{raw.subject}\n\n{raw.body}" if raw.subject else raw.body
    enquiry = NormalizedEnquiry(
        enquiry_id=str(uuid.uuid4())[:8],
        channel=raw.channel,
        sender_name=raw.sender_name,
        sender_email=raw.sender_email,
        text=text,
        received_at=raw.received_at,
    )
    return {"enquiry": enquiry}


def triage_node(state: EnquiryState) -> dict:
    try:
        return {"triage": llm.triage_enquiry(state["enquiry"])}
    except Exception as exc:  # noqa: BLE001 - failure path is intentional
        return {"error": str(exc)}


def validate_node(state: EnquiryState) -> dict:
    if state.get("error"):
        # LLM failed entirely: fail safe by routing to a human via clarification
        # rather than guessing a category or dropping the enquiry.
        return {"route": "clarify"}
    triage = state["triage"]
    result = validation_mod.validate_triage(state["enquiry"], triage)
    route = validation_mod.decide_route(triage, result)
    return {"validation": result, "route": route}


def route_selector(state: EnquiryState) -> str:
    return state["route"]


def archive_node(state: EnquiryState) -> dict:
    return {}


def clarify_node(state: EnquiryState) -> dict:
    enquiry = state["enquiry"]
    validation = state.get("validation")
    missing = validation.missing_fields if validation else ["unable to classify enquiry"]
    try:
        draft = llm.draft_clarification(enquiry, missing)
    except Exception:  # noqa: BLE001 - deterministic fallback keeps the workflow moving
        draft = (
            f"Hi {enquiry.sender_name or 'there'},\n\n"
            f"Thanks for reaching out. Could you confirm the following so we can help: "
            f"{', '.join(missing)}?\n\nBest regards,\nThe Team"
        )
    return {"draft": draft}


def duplicate_check_node(state: EnquiryState) -> dict:
    action = crm_mod.crm.process(state["enquiry"], state["triage"])
    return {"crm_action": action}


def draft_response_node(state: EnquiryState) -> dict:
    try:
        draft = llm.draft_response(state["enquiry"], state["triage"], state["crm_action"])
    except Exception:  # noqa: BLE001 - deterministic fallback keeps the workflow moving
        enquiry = state["enquiry"]
        draft = (
            f"Hi {enquiry.sender_name or 'there'},\n\n"
            f"Thanks for contacting us — reference {state['crm_action'].record_id}. "
            f"We'll follow up shortly.\n\nBest regards,\nThe Team"
        )
    return {"draft": draft}


def human_approval_node(state: EnquiryState) -> dict:
    enquiry = state["enquiry"]
    summary = f"Enquiry {enquiry.enquiry_id} ({state['route']})\n\n{state.get('draft', '')}"
    approval = approval_mod.request_human_approval(summary=summary)
    return {"approval": approval}


def audit_node(state: EnquiryState) -> dict:
    enquiry = state["enquiry"]
    triage = state.get("triage")
    crm_action = state.get("crm_action")
    approval = state.get("approval")
    validation = state.get("validation")
    audit.record_audit_event(
        enquiry_id=enquiry.enquiry_id,
        route=state["route"],
        category=triage.category.value if triage else None,
        confidence=triage.confidence if triage else None,
        is_duplicate=crm_action.is_duplicate if crm_action else None,
        crm_record_id=crm_action.record_id if crm_action else None,
        approved=approval.approved if approval else None,
        approver=approval.approver if approval else None,
        details={
            "error": state.get("error"),
            "missing_fields": validation.missing_fields if validation else None,
            "draft": state.get("draft"),
        },
    )
    return {}


def build_graph():
    graph = StateGraph(EnquiryState)
    graph.add_node("normalize", normalize_node)
    graph.add_node("triage", triage_node)
    graph.add_node("validate", validate_node)
    graph.add_node("archive", archive_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("duplicate_check", duplicate_check_node)
    graph.add_node("draft_response", draft_response_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("audit", audit_node)

    graph.set_entry_point("normalize")
    graph.add_edge("normalize", "triage")
    graph.add_edge("triage", "validate")
    graph.add_conditional_edges(
        "validate",
        route_selector,
        {
            "archive": "archive",
            "clarify": "clarify",
            "sales": "duplicate_check",
            "support": "duplicate_check",
        },
    )
    graph.add_edge("archive", "audit")
    graph.add_edge("clarify", "human_approval")
    graph.add_edge("duplicate_check", "draft_response")
    graph.add_edge("draft_response", "human_approval")
    graph.add_edge("human_approval", "audit")
    graph.add_edge("audit", END)

    return graph.compile()
