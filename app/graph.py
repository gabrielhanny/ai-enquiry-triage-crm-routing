"""LangGraph state and workflow wiring.

    Normalize -> Triage (LLM) -> Validate (deterministic) -> recommend_owner
        -> archive                                   [junk]
        -> clarify -> human_approval -> apply_crm_action  [incomplete / low confidence, LLM responded]
        -> queue_for_retry                            [LLM/API unavailable — see below]
        -> analyze_crm -> draft_response -> human_approval -> apply_crm_action
               [sales / support / technical / operations / infrastructure / other]
    -> audit (always, exactly once per enquiry)

If the LLM call itself fails (outage, timeout, etc.), triage_node never
fabricates a TriageResult. validate_node routes that case to `queued`,
distinct from `clarify` (which is for a TriageResult the LLM *did* produce,
just incomplete or low-confidence). queue_for_retry_node records the raw
enquiry in the ai_queue SQLite table (app/ai_queue.py) and goes straight to
audit — no draft, no approval, no CRM action is attempted while AI-dependent
understanding is unavailable. A later retry (run_dataset.py --retry)
re-invokes this same graph with the queued RawEnquiry; if the LLM has come
back, it proceeds through the normal path including human_approval before
any CRM mutation, exactly like a fresh enquiry.

analyze_crm is READ-ONLY: it matches the enquiry against a pool of candidate
CRM records (pre-existing rows loaded from crm.csv, plus records already
created in this run), flags any factual conflicts against the best match,
and decides a *recommended* action — but does not touch CRM state. It never
forces an uncertain match into a decision — see app/crm_match.py.

apply_crm_action is the ONLY node allowed to mutate the (mock) CRM, and it
runs after human_approval. If approval was withheld, no mutation happens.
This ordering is deliberate: consequential CRM writes (creating a lead or
ticket, or attaching an enquiry to an existing record) must never happen
before a human has had the chance to approve or reject the draft.
"""
from __future__ import annotations

import uuid
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from . import ai_queue
from . import approval as approval_mod
from . import audit
from . import crm as crm_mod
from . import crm_match
from . import llm
from . import staff as staff_mod
from . import validation as validation_mod
from .models import (
    ApprovalRecord,
    Conflict,
    CRMAction,
    CRMRecord,
    EnquiryCategory,
    MatchResult,
    NormalizedEnquiry,
    OwnerRecommendation,
    PENDING_CRM_RECORD_ID,
    RawEnquiry,
    StaffMember,
    TriageResult,
    ValidationResult,
)


class EnquiryState(TypedDict, total=False):
    raw: RawEnquiry
    source_email_id: Optional[str]
    crm_reference: list[CRMRecord]
    staff_directory: list[StaffMember]
    enquiry: NormalizedEnquiry
    triage: TriageResult
    validation: ValidationResult
    route: str
    owner: OwnerRecommendation
    match_result: MatchResult
    conflicts: list[Conflict]
    recommended_action: CRMAction
    crm_action: CRMAction
    draft: str
    approval: ApprovalRecord
    error: Optional[str]
    ai_status: str  # "available" | "unavailable"
    queue_id: Optional[int]  # set when this invocation newly queues the enquiry
    retry_of_queue_id: Optional[int]  # set by the caller when re-invoking a queued item


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
        return {"triage": llm.triage_enquiry(state["enquiry"]), "ai_status": "available"}
    except Exception as exc:  # noqa: BLE001 - failure path is intentional
        return {"error": str(exc), "ai_status": "unavailable"}


def validate_node(state: EnquiryState) -> dict:
    if state.get("error"):
        # The LLM call itself failed (outage, timeout, etc.) — no TriageResult
        # exists to validate. This is distinct from "clarify": there, the LLM
        # responded but the enquiry was incomplete/low-confidence. Here, AI
        # understanding is simply unavailable, so the enquiry is queued for
        # retry rather than fabricated or treated as an ordinary clarification.
        return {"route": "queued"}
    # Deterministic category safeguard (e.g. a customer's own contact-detail
    # correction must never be routed as `infrastructure`) runs before
    # validation/routing so every downstream consumer — owner recommendation,
    # drafting, audit — sees the corrected category consistently.
    triage = validation_mod.apply_category_safeguards(state["enquiry"], state["triage"])
    result = validation_mod.validate_triage(state["enquiry"], triage)
    route = validation_mod.decide_route(triage, result)
    return {"triage": triage, "validation": result, "route": route}


def route_selector(state: EnquiryState) -> str:
    return state["route"]


def recommend_owner_node(state: EnquiryState) -> dict:
    """No-ops when no staff directory was supplied (e.g. Test 1 scenarios),
    so this is a pure addition with no effect on existing behavior."""
    staff = state.get("staff_directory")
    triage = state.get("triage")
    if not staff or not triage:
        return {}
    return {"owner": staff_mod.recommend_owner(triage.category.value, staff)}


def archive_node(state: EnquiryState) -> dict:
    return {}


def queue_for_retry_node(state: EnquiryState) -> dict:
    """AI-dependent work (classification, extraction, drafting) could not
    be completed because the LLM was unavailable. Records the raw enquiry
    in the ai_queue SQLite table for later retry — never fabricates a
    TriageResult, and performs no draft, approval, or CRM action here."""
    enquiry = state["enquiry"]
    queue_id = ai_queue.enqueue(
        enquiry_id=enquiry.enquiry_id,
        source_email_id=state.get("source_email_id"),
        raw=state["raw"],
        error=state.get("error"),
    )
    return {"queue_id": queue_id}


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


def analyze_crm_node(state: EnquiryState) -> dict:
    """READ-ONLY. Matches against a pool of candidate records (pre-existing
    CRM rows plus records already created this run) and decides what action
    *would* be taken, without performing it. No CRM state is mutated here —
    see apply_crm_action_node, which runs after human approval."""
    enquiry = state["enquiry"]
    triage = state["triage"]

    crm_reference = state.get("crm_reference") or []
    candidates = [crm_match.from_crm_record(r) for r in crm_reference]
    candidates += [crm_match.from_mock_record(r) for r in crm_mod.crm.all_records()]

    match_result = crm_match.match_enquiry(enquiry, triage, candidates)
    conflicts: list[Conflict] = []

    if match_result.status in ("likely_match", "possible_match"):
        best = match_result.best
        matched_candidate = crm_match.find_candidate(best.record_id, candidates)
        conflicts = crm_match.detect_conflicts(enquiry, triage, matched_candidate, enquiry.enquiry_id)
        recommended_action = CRMAction(
            action="attach_to_existing_crm_record" if match_result.status == "likely_match" else "flag_for_review",
            record_id=best.record_id,
            is_duplicate=match_result.status == "likely_match",
            details={"match_score": best.score, "signals": best.signals},
        )
    else:
        # Real id is only assigned when the record is actually created,
        # post-approval — see apply_crm_action_node.
        is_sales = triage.category == EnquiryCategory.SALES
        recommended_action = CRMAction(
            action="create_lead" if is_sales else "create_ticket",
            record_id=PENDING_CRM_RECORD_ID,
            is_duplicate=False,
            details={"category": triage.category.value},
        )

    return {"recommended_action": recommended_action, "match_result": match_result, "conflicts": conflicts}


def apply_crm_action_node(state: EnquiryState) -> dict:
    """The ONLY node allowed to mutate the (mock) CRM. Runs after human
    approval. No-ops (returns {}) when there is nothing to apply — e.g. the
    clarify path never reaches analyze_crm, so there's no recommended
    action here. If approval was withheld, records that as-is without
    mutating anything."""
    recommended = state.get("recommended_action")
    if recommended is None:
        return {}

    approval = state.get("approval")
    if not approval or not approval.approved:
        return {
            "crm_action": CRMAction(
                action="not_applied",
                record_id=recommended.record_id,
                is_duplicate=False,
                details={"reason": "approval withheld or missing", "recommended_action": recommended.action},
            )
        }

    enquiry = state["enquiry"]
    triage = state["triage"]

    if recommended.action in ("attach_to_existing_crm_record", "flag_for_review"):
        # Mutates only if the matched record was itself created earlier in
        # this run; a no-op for a static external crm.csv row, which has no
        # local mutable copy to update.
        crm_mod.crm.attach(recommended.record_id, enquiry.enquiry_id)
        return {"crm_action": recommended}

    # create_lead / create_ticket: the record is only actually created now.
    final_action = crm_mod.crm.process(enquiry, triage)
    return {"crm_action": final_action}


def draft_response_node(state: EnquiryState) -> dict:
    """Drafts against the *recommended* CRM action (pre-approval) — the
    real record_id, if any, isn't assigned until apply_crm_action_node runs
    after approval."""
    recommended = state["recommended_action"]
    try:
        draft = llm.draft_response(state["enquiry"], state["triage"], recommended)
    except Exception:  # noqa: BLE001 - deterministic fallback keeps the workflow moving
        enquiry = state["enquiry"]
        reference = (
            f" — reference {recommended.record_id}" if recommended.record_id != PENDING_CRM_RECORD_ID else ""
        )
        draft = (
            f"Hi {enquiry.sender_name or 'there'},\n\n"
            f"Thanks for contacting us{reference}. "
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
    recommended = state.get("recommended_action")
    approval = state.get("approval")
    validation = state.get("validation")
    match_result = state.get("match_result")
    conflicts = state.get("conflicts") or []
    owner = state.get("owner")
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
            "source_email_id": state.get("source_email_id"),
            "error": state.get("error"),
            "ai_status": state.get("ai_status", "available"),
            "degraded_mode": state.get("route") == "queued",
            "queue_id": state.get("queue_id"),
            "retry_of_queue_id": state.get("retry_of_queue_id"),
            "reasoning": triage.reasoning if triage else None,
            "missing_fields": validation.missing_fields if validation else None,
            "draft": state.get("draft"),
            "match_status": match_result.status if match_result else None,
            "match_candidates": [c.model_dump() for c in match_result.candidates] if match_result else None,
            "match_notes": match_result.notes if match_result else None,
            "conflicts": [c.model_dump() for c in conflicts],
            "recommended_action": recommended.model_dump() if recommended else None,
            "owner": owner.owner if owner else None,
            "owner_reasoning": owner.reasoning if owner else None,
        },
    )
    return {}


def build_graph():
    graph = StateGraph(EnquiryState)
    graph.add_node("normalize", normalize_node)
    graph.add_node("triage", triage_node)
    graph.add_node("validate", validate_node)
    graph.add_node("recommend_owner", recommend_owner_node)
    graph.add_node("archive", archive_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("queue_for_retry", queue_for_retry_node)
    graph.add_node("analyze_crm", analyze_crm_node)
    graph.add_node("draft_response", draft_response_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("apply_crm_action", apply_crm_action_node)
    graph.add_node("audit", audit_node)

    graph.set_entry_point("normalize")
    graph.add_edge("normalize", "triage")
    graph.add_edge("triage", "validate")
    graph.add_edge("validate", "recommend_owner")
    graph.add_conditional_edges(
        "recommend_owner",
        route_selector,
        {
            "archive": "archive",
            "clarify": "clarify",
            "queued": "queue_for_retry",
            "sales": "analyze_crm",
            "support": "analyze_crm",
            "technical": "analyze_crm",
            "operations": "analyze_crm",
            "infrastructure": "analyze_crm",
            "other": "analyze_crm",
        },
    )
    graph.add_edge("archive", "audit")
    graph.add_edge("clarify", "human_approval")
    graph.add_edge("queue_for_retry", "audit")
    graph.add_edge("analyze_crm", "draft_response")
    graph.add_edge("draft_response", "human_approval")
    # apply_crm_action is shared by both approval-gated paths; it no-ops
    # when there's no recommended_action (the clarify path).
    graph.add_edge("human_approval", "apply_crm_action")
    graph.add_edge("apply_crm_action", "audit")
    graph.add_edge("audit", END)

    return graph.compile()
