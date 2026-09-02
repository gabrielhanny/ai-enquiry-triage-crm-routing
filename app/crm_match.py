"""Deterministic CRM matching and conflict detection.

Matches an enquiry against a pool of candidate records — both pre-existing
CRM rows (crm.csv) and records already created in this run by the mock CRM —
using multiple weak signals (email, phone, company name, contact name,
location). This never trusts a single fuzzy signal alone: an exact email
match is treated as a strong identity signal, but everything else has to
accumulate before a match is called "likely". Ambiguous cases are reported,
not resolved, so a human can decide.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .models import Conflict, CRMRecord, MatchCandidate, MatchResult, NormalizedEnquiry, TriageResult

LIKELY_THRESHOLD = 0.75
POSSIBLE_THRESHOLD = 0.35

_SUFFIX_RE = re.compile(r"\b(pty\.?\s*ltd\.?|limited|ltd\.?|inc\.?)\b", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")
_NON_DIGIT_RE = re.compile(r"\D")


def normalize_company(name: str | None) -> str:
    if not name:
        return ""
    cleaned = _SUFFIX_RE.sub("", name.lower())
    cleaned = _NON_ALNUM_RE.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_phone(phone: str | None) -> str:
    return _NON_DIGIT_RE.sub("", phone or "")


def _email_domain(email: str) -> str:
    return email.split("@", 1)[1].lower() if "@" in email else ""


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


@dataclass
class Candidate:
    """Lightweight comparable shape shared by external CRM rows and
    in-memory mock-CRM records, so both can be matched the same way."""

    record_id: str
    company: str = ""
    contact: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""


def from_crm_record(record: CRMRecord) -> Candidate:
    return Candidate(record.id, record.company, record.contact, record.email, record.phone, record.location)


def from_mock_record(record: dict) -> Candidate:
    return Candidate(
        record["record_id"],
        record.get("company", ""),
        record.get("contact", ""),
        record.get("email", ""),
        record.get("phone", ""),
        record.get("location", ""),
    )


def score_candidate(enquiry: NormalizedEnquiry, triage: TriageResult, candidate: Candidate) -> tuple[float, list[str]]:
    score = 0.0
    signals: list[str] = []

    enquiry_email = (enquiry.sender_email or "").lower()
    candidate_email = candidate.email.lower()
    if enquiry_email and candidate_email and enquiry_email == candidate_email:
        score = max(score, 0.9)
        signals.append("exact email match")
    elif enquiry_email and candidate_email and _email_domain(enquiry_email) == _email_domain(candidate_email):
        score += 0.45
        signals.append("email domain match")

    enquiry_phone = normalize_phone(triage.phone)
    candidate_phone = normalize_phone(candidate.phone)
    if enquiry_phone and candidate_phone and enquiry_phone == candidate_phone:
        score += 0.35
        signals.append("phone match")

    company_ratio = _ratio(normalize_company(triage.company_name), normalize_company(candidate.company))
    if company_ratio >= 0.85:
        score += 0.5
        signals.append("company name match")
    elif company_ratio >= 0.6:
        score += 0.25
        signals.append("company name similar")

    contact_ratio = _ratio((triage.contact_name or "").lower(), candidate.contact.lower())
    if contact_ratio >= 0.8:
        score += 0.2
        signals.append("contact name match")

    enquiry_location = (triage.location or "").lower()
    candidate_location = candidate.location.lower()
    if enquiry_location and candidate_location and (
        enquiry_location in candidate_location or candidate_location in enquiry_location
    ):
        score += 0.1
        signals.append("location match")

    return min(score, 1.0), signals


def match_enquiry(enquiry: NormalizedEnquiry, triage: TriageResult, candidates: list[Candidate]) -> MatchResult:
    scored = []
    for candidate in candidates:
        score, signals = score_candidate(enquiry, triage, candidate)
        if score > 0:
            scored.append((round(score, 2), candidate, signals))
    scored.sort(key=lambda item: item[0], reverse=True)

    match_candidates = [
        MatchCandidate(record_id=c.record_id, company=c.company, score=score, signals=signals)
        for score, c, signals in scored[:5]
    ]

    if not match_candidates:
        return MatchResult(status="no_match", candidates=[], notes=[])

    top_score = match_candidates[0].score
    if top_score >= LIKELY_THRESHOLD:
        status = "likely_match"
    elif top_score >= POSSIBLE_THRESHOLD:
        status = "possible_match"
    else:
        status = "no_match"

    notes: list[str] = []
    if (
        len(match_candidates) >= 2
        and (match_candidates[0].score - match_candidates[1].score) < 0.15
        and match_candidates[1].score >= POSSIBLE_THRESHOLD
    ):
        notes.append(
            f"{match_candidates[0].record_id} and {match_candidates[1].record_id} scored closely "
            f"({match_candidates[0].score} vs {match_candidates[1].score}) - "
            "possibly the same organisation recorded twice in the CRM."
        )

    return MatchResult(status=status, candidates=match_candidates, notes=notes)


def find_candidate(record_id: str, candidates: list[Candidate]) -> Candidate | None:
    return next((c for c in candidates if c.record_id == record_id), None)


def detect_conflicts(
    enquiry: NormalizedEnquiry, triage: TriageResult, candidate: Candidate | None, source_id: str
) -> list[Conflict]:
    """Compares new evidence to a matched record's stored values. Only
    flags a conflict when BOTH sides have a value and they disagree — a
    blank CRM field being filled in is new information, not a conflict."""
    if candidate is None:
        return []
    conflicts: list[Conflict] = []

    enquiry_email = enquiry.sender_email
    if enquiry_email and candidate.email and enquiry_email.lower() != candidate.email.lower():
        conflicts.append(
            Conflict(
                field="email",
                existing_value=candidate.email,
                new_value=enquiry_email,
                source=source_id,
                note="Enquiry email differs from the CRM record on file.",
            )
        )

    if triage.phone and candidate.phone and normalize_phone(triage.phone) != normalize_phone(candidate.phone):
        conflicts.append(
            Conflict(
                field="phone",
                existing_value=candidate.phone,
                new_value=triage.phone,
                source=source_id,
                note="Enquiry phone number differs from the CRM record on file.",
            )
        )

    if triage.company_name and candidate.company:
        ratio = _ratio(normalize_company(triage.company_name), normalize_company(candidate.company))
        if ratio < 0.5:
            conflicts.append(
                Conflict(
                    field="company",
                    existing_value=candidate.company,
                    new_value=triage.company_name,
                    source=source_id,
                    note="Enquiry company name differs substantially from the matched CRM record.",
                )
            )

    return conflicts
