import pytest
from pydantic import ValidationError

from app.models import EnquiryCategory, TriageResult


def test_triage_result_accepts_valid_confidence():
    result = TriageResult(category=EnquiryCategory.SALES, confidence=0.9, reasoning="clear intent")
    assert result.category == EnquiryCategory.SALES
    assert result.missing_information == []


def test_triage_result_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        TriageResult(category=EnquiryCategory.SALES, confidence=1.5, reasoning="bad")


def test_triage_result_rejects_unknown_category():
    with pytest.raises(ValidationError):
        TriageResult(category="not_a_category", confidence=0.5, reasoning="bad")
