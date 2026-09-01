"""Mock CRM adapter.

Simulates lead/ticket creation and duplicate detection. This is an in-memory
stand-in only — it never touches a real CRM, never persists real customer
data, and exists purely to demonstrate the duplicate-check and CRM-action
steps of the workflow.
"""
from __future__ import annotations

import itertools

from .models import CRMAction, EnquiryCategory, NormalizedEnquiry, TriageResult


class MockCRM:
    def __init__(self) -> None:
        self._records: dict[str, list[dict]] = {}
        self._counter = itertools.count(1)

    def _next_id(self, prefix: str) -> str:
        return f"{prefix}-{next(self._counter):04d}"

    def find_open_record(self, email: str, category: EnquiryCategory) -> dict | None:
        for record in self._records.get(email, []):
            if record["category"] == category.value and record["status"] == "open":
                return record
        return None

    def process(self, enquiry: NormalizedEnquiry, triage: TriageResult) -> CRMAction:
        email = enquiry.sender_email or "unknown@unknown"
        existing = self.find_open_record(email, triage.category)
        if existing is not None:
            existing["enquiry_ids"].append(enquiry.enquiry_id)
            return CRMAction(
                action="attach_to_existing",
                record_id=existing["record_id"],
                is_duplicate=True,
                details={"category": triage.category.value, "enquiry_ids": existing["enquiry_ids"]},
            )

        is_sales = triage.category == EnquiryCategory.SALES
        record_id = self._next_id("LEAD" if is_sales else "TCKT")
        self._records.setdefault(email, []).append(
            {
                "record_id": record_id,
                "email": email,
                "category": triage.category.value,
                "status": "open",
                "enquiry_ids": [enquiry.enquiry_id],
            }
        )
        return CRMAction(
            action="create_lead" if is_sales else "create_ticket",
            record_id=record_id,
            is_duplicate=False,
            details={"category": triage.category.value},
        )


# Module-level singleton so a demo run can accumulate CRM state across
# scenarios (e.g. to demonstrate duplicate detection on a repeat enquiry).
crm = MockCRM()
