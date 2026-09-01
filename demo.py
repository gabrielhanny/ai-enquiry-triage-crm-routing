"""Runs the four locked demo scenarios end-to-end through the LangGraph workflow.

Safe by default: no real emails are sent, no real CRM is touched, and every
enquiry produces exactly one row in the SQLite audit trail (data/audit.db).
"""
from app.graph import build_graph
from app.scenarios import SCENARIOS


def run_scenario(name: str, raw_enquiry, workflow) -> dict:
    print(f"\n{'=' * 70}\nSCENARIO: {name}\n{'=' * 70}")
    result = workflow.invoke({"raw": raw_enquiry})

    triage = result.get("triage")
    print(f"Route taken   : {result.get('route')}")
    if triage:
        print(f"Category      : {triage.category.value} (confidence={triage.confidence:.2f})")
    if result.get("validation"):
        print(f"Validation    : {result['validation']}")
    if result.get("crm_action"):
        ca = result["crm_action"]
        print(f"CRM action    : {ca.action} -> {ca.record_id} (duplicate={ca.is_duplicate})")
    if result.get("approval"):
        print(f"Approval      : approved={result['approval'].approved} by {result['approval'].approver}")
    if result.get("error"):
        print(f"LLM error     : {result['error']}")
    return result


def main() -> None:
    workflow = build_graph()

    for name, raw in SCENARIOS:
        run_scenario(name, raw, workflow)

    # Bonus: re-submit the first (sales) enquiry to demonstrate the
    # duplicate-check branch of the workflow.
    dup_name, dup_raw = SCENARIOS[0]
    run_scenario(f"{dup_name} (RESUBMIT - duplicate check)", dup_raw, workflow)

    print(f"\n{'=' * 70}\nAll scenarios complete. Audit trail written to data/audit.db\n{'=' * 70}")


if __name__ == "__main__":
    main()
