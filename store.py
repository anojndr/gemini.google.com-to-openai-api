# Copyright 2026 gem2oai contributors.
"""SQLite persistence: one shared schema for sessions, aliases, responses, files.

Single file (config.DB_PATH), WAL mode, created on first connect. The
conversations.SessionStore and content.UploadStore each hold their own
sqlite3 handle to this file and write through on every mutation, so a
restart loses no conversations, response chains, or uploaded files.
"""

from __future__ import annotations

import os
import sqlite3

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS sessions (
    key TEXT PRIMARY KEY,
    account INTEGER,
    metadata TEXT NOT NULL DEFAULT '[]',
    norm TEXT NOT NULL DEFAULT '[]',
    updated_at INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS aliases (
    alias TEXT PRIMARY KEY,
    target TEXT NOT NULL,
    updated_at INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS responses (
    id TEXT PRIMARY KEY,
    body TEXT NOT NULL,
    updated_at INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    data BLOB NOT NULL,
    mime TEXT NOT NULL DEFAULT 'application/octet-stream',
    filename TEXT NOT NULL DEFAULT 'upload.bin',
    purpose TEXT NOT NULL DEFAULT 'assistants',
    created_at INTEGER NOT NULL DEFAULT 0
);
"""


def connect(path: os.PathLike[str] | str) -> sqlite3.Connection:
    """Open the state database at path, creating the schema on first use."""
    db = sqlite3.connect(os.fspath(path), timeout=30.0)
    db.execute("PRAGMA journal_mode=WAL;")
    db.execute("PRAGMA busy_timeout=5000;")
    db.executescript(_SCHEMA)
    db.commit()
    return db
