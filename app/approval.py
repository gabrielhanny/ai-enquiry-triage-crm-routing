"""Human-in-the-loop approval gate.

Stands in for a real reviewer clicking approve/reject in a UI or queue. The
LLM never has approval authority. This prototype runs in safe/dry-run mode:
the draft is printed for review and auto-approved, but nothing is ever sent
to a real customer, CRM, or email system.
"""
from __future__ import annotations

from .models import ApprovalRecord


def request_human_approval(*, summary: str, approver: str = "demo-reviewer") -> ApprovalRecord:
    print("\n--- HUMAN APPROVAL REQUIRED (dry-run: auto-approved) ---")
    print(summary)
    print("--- end of draft ---\n")
    return ApprovalRecord(approved=True, approver=approver, note="auto-approved in demo dry-run mode")
