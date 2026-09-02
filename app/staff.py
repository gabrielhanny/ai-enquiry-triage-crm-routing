"""Deterministic staff/owner routing.

The LLM classifies the enquiry; this module — not the LLM — decides who
should own it, by scoring each staff member's stated area of ownership
(from staff_directory.json) against a fixed keyword list per category. If
nothing matches, the enquiry is left unassigned rather than guessed at.
"""
from __future__ import annotations

from .models import OwnerRecommendation, StaffMember

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "sales": ["commercial", "partnership", "partnerships"],
    "support": ["operational", "administration", "logistics", "scheduling"],
    "technical": ["systems", "infrastructure"],
    "operations": ["scheduling", "administration", "logistics", "operational"],
    "infrastructure": ["crm", "systems", "data", "workflows", "infrastructure"],
    "other": ["marketing", "website", "growth", "inbound"],
    "junk": [],
}


def recommend_owner(category: str, staff: list[StaffMember]) -> OwnerRecommendation:
    keywords = CATEGORY_KEYWORDS.get(category, [])
    if not keywords or not staff:
        return OwnerRecommendation(
            owner=None,
            confidence=0.0,
            reasoning=f"No staff ownership keywords defined for category '{category}'; needs manual triage.",
        )

    best_member: StaffMember | None = None
    best_score = 0
    for member in staff:
        owns_lower = member.owns.lower()
        score = sum(1 for keyword in keywords if keyword in owns_lower)
        if score > best_score:
            best_member = member
            best_score = score

    if best_member is None:
        return OwnerRecommendation(
            owner=None,
            confidence=0.0,
            reasoning=f"No staff member's stated ownership matches category '{category}'; needs manual triage.",
        )

    confidence = round(min(1.0, best_score / len(keywords)), 2)
    return OwnerRecommendation(
        owner=best_member.name,
        confidence=confidence,
        reasoning=f'Matched category "{category}" to {best_member.name} ({best_member.role}), '
        f'who owns: "{best_member.owns}".',
    )
