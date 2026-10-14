from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_path() -> Path:
    return Path(os.getenv("ASKAI_DB_PATH", "backend/askai.sqlite3"))


@contextmanager
def connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize() -> None:
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS configurations (
                id TEXT PRIMARY KEY,
                candidate_token TEXT NOT NULL UNIQUE,
                dashboard_token TEXT NOT NULL UNIQUE,
                settings_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                access_token TEXT NOT NULL UNIQUE,
                configuration_id TEXT REFERENCES configurations(id),
                candidate_name TEXT NOT NULL,
                role TEXT NOT NULL,
                settings_json TEXT NOT NULL,
                cv_analysis_json TEXT NOT NULL,
                current_question_json TEXT,
                difficulty TEXT NOT NULL,
                status TEXT NOT NULL,
                report_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                position INTEGER NOT NULL,
                question_json TEXT NOT NULL,
                answer TEXT NOT NULL,
                evaluation_json TEXT NOT NULL,
                UNIQUE(session_id, position)
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_configuration ON sessions(configuration_id);
            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, position);
            """
        )


def obj(row):
    return dict(row) if row else None


def unpack_session(row):
    item = obj(row)
    if item is None:
        return None
    for field in ("settings_json", "cv_analysis_json", "current_question_json", "report_json"):
        item[field[:-5]] = json.loads(item[field]) if item[field] else None
        del item[field]
    return item
