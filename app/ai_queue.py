"""SQLite-backed queue for AI-dependent work deferred by an LLM outage.

Same approach as app/audit.py: a lazy-path table in the existing SQLite
file, no external infrastructure (no Redis/Celery). Enqueuing is idempotent
per enquiry/source identifier — re-attempting the same enquiry while it's
still queued just bumps the attempt count rather than creating a second
row, and an item already marked 'done' is left alone.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import RawEnquiry


def _get_db_path() -> Path:
    # Same DB file/env var as app/audit.py, so tests that isolate
    # AUDIT_DB_PATH automatically isolate the queue too.
    return Path(os.getenv("AUDIT_DB_PATH", "data/audit.db"))


def get_connection() -> sqlite3.Connection:
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idempotency_key TEXT NOT NULL UNIQUE,
            enquiry_id TEXT NOT NULL,
            source_email_id TEXT,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 1,
            last_error TEXT,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return conn


def enqueue(*, enquiry_id: str, source_email_id: str | None, raw: RawEnquiry, error: str | None) -> int:
    """Idempotent on (source_email_id or enquiry_id): re-enqueuing the same
    key while it's still 'queued' bumps attempts/last_error on the existing
    row instead of inserting a duplicate. A key already marked 'done' is
    left untouched — enqueue never resurrects completed work."""
    idempotency_key = source_email_id or enquiry_id
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        with conn:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO ai_queue
                        (idempotency_key, enquiry_id, source_email_id, status,
                         attempts, last_error, payload, created_at, updated_at)
                    VALUES (?, ?, ?, 'queued', 1, ?, ?, ?, ?)
                    """,
                    (idempotency_key, enquiry_id, source_email_id, error, raw.model_dump_json(), now, now),
                )
                return cur.lastrowid
            except sqlite3.IntegrityError:
                row_id, status = conn.execute(
                    "SELECT id, status FROM ai_queue WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if status != "done":
                    conn.execute(
                        """
                        UPDATE ai_queue
                        SET status = 'queued', attempts = attempts + 1, last_error = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (error, now, row_id),
                    )
                return row_id
    finally:
        conn.close()


def list_queued() -> list[dict]:
    return _select("SELECT * FROM ai_queue WHERE status = 'queued' ORDER BY id")


def list_all() -> list[dict]:
    return _select("SELECT * FROM ai_queue ORDER BY id")


def _select(sql: str) -> list[dict]:
    conn = get_connection()
    try:
        cur = conn.execute(sql)
        columns = [c[0] for c in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def mark_done(queue_id: int) -> None:
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE ai_queue SET status = 'done', updated_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), queue_id),
            )
    finally:
        conn.close()
