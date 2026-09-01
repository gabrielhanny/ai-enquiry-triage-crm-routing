import pytest

from app import crm as crm_module


@pytest.fixture(autouse=True)
def isolated_audit_db(tmp_path, monkeypatch):
    """Point the audit trail at a throwaway file so tests never touch data/audit.db."""
    monkeypatch.setenv("AUDIT_DB_PATH", str(tmp_path / "test_audit.db"))


@pytest.fixture(autouse=True)
def reset_mock_crm():
    """Give each test a fresh in-memory CRM so duplicate-detection state doesn't leak."""
    crm_module.crm = crm_module.MockCRM()
