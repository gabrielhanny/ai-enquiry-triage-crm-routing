"""SQLite-backed audit trail.

Every enquiry that passes through the workflow gets exactly one audit row,
regardless of which path it took. This is the durable, deterministic record
of what happened — it does not depend on the LLM being available or correct.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _get_db_path() -> Path:
    # Read lazily (not at import time) so tests can point this at a temp file.
    return Path(os.getenv("AUDIT_DB_PATH", "data/audit.db"))


def get_connection() -> sqlite3.Connection:
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            enquiry_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            route TEXT NOT NULL,
            category TEXT,
            confidence REAL,
            is_duplicate INTEGER,
            crm_record_id TEXT,
            approved INTEGER,
            approver TEXT,
            details TEXT
        )
        """
    )
    return conn


def record_audit_event(
    *,
    enquiry_id: str,
    route: str,
    category: str | None = None,
    confidence: float | None = None,
    is_duplicate: bool | None = None,
    crm_record_id: str | None = None,
    approved: bool | None = None,
    approver: str | None = None,
    details: dict | None = None,
) -> int:
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO audit_log
                    (enquiry_id, timestamp, route, category, confidence,
                     is_duplicate, crm_record_id, approved, approver, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    enquiry_id,
                    datetime.now(timezone.utc).isoformat(),
                    route,
                    category,
                    confidence,
                    None if is_duplicate is None else int(is_duplicate),
                    crm_record_id,
                    None if approved is None else int(approved),
                    approver,
                    json.dumps(details or {}),
                ),
            )
            row_id = cur.lastrowid
    finally:
        conn.close()
    return row_id


def fetch_all_events() -> list[dict]:
    conn = get_connection()
    try:
        cur = conn.execute("SELECT * FROM audit_log ORDER BY id")
        columns = [c[0] for c in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        conn.close()
