"""Test 2: ingest the supplied fictional dataset and run every enquiry
through the same LangGraph workflow used in Test 1, then provide simple CLI
inspection of the results.

Safe by default, same as demo.py: no real CRM, email, or messaging
integration. crm.csv is a static local fixture, not a live system.

Usage:
    python run_dataset.py                 # table of all enquiries
    python run_dataset.py --id E005       # full detail for one enquiry
    python run_dataset.py --json          # structured dump of all results
    python run_dataset.py --queue         # list AI-dependent work awaiting retry
    python run_dataset.py --retry         # retry queued AI-dependent work now
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys

from app import ai_queue, ingest
from app.graph import build_graph
from app.models import RawEnquiry

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def run_all(quiet: bool = False) -> list[dict]:
    """quiet=True suppresses the human-approval console prints (which would
    otherwise interleave with --json output on the same stdout stream)."""
    staff = ingest.load_staff_directory()
    crm_records = ingest.load_crm_records()
    emails = ingest.load_emails()
    workflow = build_graph()

    results = []
    sink = io.StringIO() if quiet else None
    for email in emails:
        raw = ingest.email_to_raw_enquiry(email)
        with contextlib.redirect_stdout(sink) if sink else contextlib.nullcontext():
            state = workflow.invoke(
                {
                    "raw": raw,
                    "crm_reference": crm_records,
                    "staff_directory": staff,
                    "source_email_id": email.id,
                }
            )
        results.append(state)
    return results


def print_queue() -> None:
    items = ai_queue.list_all()
    if not items:
        print("ai_queue is empty.")
        return
    header = f"{'id':<4}{'source':<8}{'status':<9}{'attempts':<9}{'updated_at':<28}last_error"
    print(header)
    print("-" * len(header))
    for item in items:
        err = (item["last_error"] or "")[:60]
        print(
            f"{item['id']:<4}{item['source_email_id'] or item['enquiry_id']:<8}"
            f"{item['status']:<9}{item['attempts']:<9}{item['updated_at']:<28}{err}"
        )


def retry_queue() -> list[dict]:
    """Retries every currently-queued item. Idempotent: an item already
    marked 'done' is never re-processed (list_queued only returns 'queued'
    rows), so re-running --retry after a successful drain is a no-op."""
    workflow = build_graph()
    crm_records = ingest.load_crm_records()
    staff = ingest.load_staff_directory()

    results = []
    queued = ai_queue.list_queued()
    if not queued:
        print("Nothing queued to retry.")
        return results

    for item in queued:
        raw = RawEnquiry.model_validate_json(item["payload"])
        state = workflow.invoke(
            {
                "raw": raw,
                "crm_reference": crm_records,
                "staff_directory": staff,
                "source_email_id": item["source_email_id"],
                "retry_of_queue_id": item["id"],
            }
        )
        label = item["source_email_id"] or item["enquiry_id"]
        if state.get("error"):
            print(f"Retry {label}: still unavailable ({state['error']})")
        else:
            ai_queue.mark_done(item["id"])
            print(f"Retry {label}: succeeded (route={state.get('route')})")
        results.append(state)
    return results


def _fmt_match(state: dict) -> str:
    match = state.get("match_result")
    if not match or not match.candidates:
        return "no_match"
    best = match.best
    return f"{match.status} -> {best.record_id} ({best.score})"


def _fmt_owner(state: dict) -> str:
    owner = state.get("owner")
    if not owner or not owner.owner:
        return "unassigned"
    return owner.owner


def print_table(results: list[dict]) -> None:
    columns = [
        ("ID", 6),
        ("AI", 5),
        ("Category", 16),
        ("Conf.", 7),
        ("Route", 16),
        ("Match", 28),
        ("Owner", 18),
        ("Conflicts", 10),
        ("Approved", 9),
    ]
    header = "".join(f"{name:<{width}}" for name, width in columns)
    print(header)
    print("-" * len(header))
    for state in results:
        triage = state.get("triage")
        approval = state.get("approval")
        ai_status = state.get("ai_status", "available")
        row = [
            state.get("source_email_id", "?"),
            "ok" if ai_status == "available" else "DOWN",
            triage.category.value if triage else "-",
            f"{triage.confidence:.2f}" if triage else "-",
            state.get("route", "-"),
            _fmt_match(state),
            _fmt_owner(state),
            str(len(state.get("conflicts") or [])),
            "yes" if approval and approval.approved else ("-" if not approval else "no"),
        ]
        print("".join(f"{str(value):<{width}}" for value, (_, width) in zip(row, columns)))


def print_detail(results: list[dict], enquiry_id: str) -> None:
    for state in results:
        if state.get("source_email_id") == enquiry_id:
            _print_one_detail(state)
            return
    print(f"No result found for {enquiry_id}")


def _print_one_detail(state: dict) -> None:
    enquiry = state["enquiry"]
    triage = state.get("triage")
    validation = state.get("validation")
    match = state.get("match_result")
    conflicts = state.get("conflicts") or []
    owner = state.get("owner")
    approval = state.get("approval")

    print(f"\n{'=' * 70}\n{state.get('source_email_id')} - {enquiry.sender_email or enquiry.sender_name}\n{'=' * 70}")
    print(f"AI status     : {state.get('ai_status', 'available')}")
    print(f"Route         : {state.get('route')}")
    if state.get("route") == "queued":
        print(f"Queued        : queue_id={state.get('queue_id')} — AI unavailable, awaiting retry")
    if triage:
        print(f"Category      : {triage.category.value} (confidence={triage.confidence:.2f})")
        print(f"Reasoning     : {triage.reasoning}")
        print(
            f"Extracted     : company={triage.company_name!r} contact={triage.contact_name!r} "
            f"phone={triage.phone!r} location={triage.location!r}"
        )
        if triage.missing_information:
            print(f"LLM-flagged missing info: {triage.missing_information}")
    if validation:
        print(f"Validation    : valid={validation.is_valid} missing={validation.missing_fields} reasons={validation.reasons}")
    if match:
        print(f"CRM match     : status={match.status}")
        for candidate in match.candidates:
            print(f"   candidate  : {candidate.record_id} ({candidate.company!r}) score={candidate.score} signals={candidate.signals}")
        for note in match.notes:
            print(f"   note       : {note}")
    for conflict in conflicts:
        print(f"Conflict      : field={conflict.field} existing={conflict.existing_value!r} new={conflict.new_value!r} source={conflict.source}")
    if owner:
        print(f"Owner         : {owner.owner or 'unassigned'} ({owner.reasoning})")
    if state.get("crm_action"):
        action = state["crm_action"]
        print(f"CRM action    : {action.action} -> {action.record_id} (duplicate={action.is_duplicate})")
    if state.get("draft"):
        print(f"Draft         :\n{state['draft']}")
    if approval:
        print(f"Approval      : approved={approval.approved} by {approval.approver}")
    if state.get("error"):
        print(f"LLM error     : {state['error']}")


def _dump_json(results: list[dict]) -> str:
    dumped = []
    for state in results:
        dumped.append(
            {
                "id": state.get("source_email_id"),
                "route": state.get("route"),
                "ai_status": state.get("ai_status", "available"),
                "queue_id": state.get("queue_id"),
                "retry_of_queue_id": state.get("retry_of_queue_id"),
                "triage": state["triage"].model_dump() if state.get("triage") else None,
                "validation": state["validation"].model_dump() if state.get("validation") else None,
                "match_result": state["match_result"].model_dump() if state.get("match_result") else None,
                "conflicts": [c.model_dump() for c in (state.get("conflicts") or [])],
                "owner": state["owner"].model_dump() if state.get("owner") else None,
                "crm_action": state["crm_action"].model_dump() if state.get("crm_action") else None,
                "approval": state["approval"].model_dump() if state.get("approval") else None,
                "error": state.get("error"),
            }
        )
    return json.dumps(dumped, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Test 2 dataset through the enquiry triage workflow.")
    parser.add_argument("--id", help="Show full detail for one enquiry id (e.g. E005)")
    parser.add_argument("--json", action="store_true", help="Dump raw results as JSON instead of a table")
    parser.add_argument("--queue", action="store_true", help="List AI-dependent work awaiting retry")
    parser.add_argument("--retry", action="store_true", help="Retry queued AI-dependent work now")
    args = parser.parse_args()

    if args.queue:
        print_queue()
        return
    if args.retry:
        retry_queue()
        print()
        print_queue()
        return

    results = run_all(quiet=args.json)

    if args.id:
        print_detail(results, args.id)
    elif args.json:
        print(_dump_json(results))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
