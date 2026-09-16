# Copyright 2026 gem2oai contributors.
"""Session store: conversation keys/aliases -> account-bound Gemini continuation.

Continuation is stored as plain Gemini metadata ([cid, rid, rcid]), so any
request can resume native server-side history with a fresh ChatSession on the
same account. Keys come from Response conversation ids, previous_response_id
chains, or content fingerprints of Chat Completions prefixes.

State is written through to SQLite (config.DB_PATH) on every mutation and
reloaded on boot, so restarts lose no conversations, aliases, or responses.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from store import connect

if TYPE_CHECKING:
    import sqlite3


def _loads_list(raw: str) -> list[Any]:
    """Parse a JSON list column, returning [] for corrupt or non-list values."""
    try:
        val = json.loads(raw)
    except ValueError:
        return []
    return list(val) if isinstance(val, list) else []


def _dumps(obj: object) -> str:
    """Encode state for SQLite TEXT columns, stringifying exotic values."""
    return json.dumps(obj, default=str)


class SessionState:
    """Per-conversation continuation state plus its serialization lock."""

    __slots__ = ("account", "lock", "metadata", "norm")

    def __init__(self) -> None:
        """Create an empty continuation state with its own lock."""
        self.account: int | None = None
        self.metadata: list[str | None] = []
        self.norm: list[str] = []
        self.lock = asyncio.Lock()


class SessionStore:
    """Conversation continuation state, persisted to SQLite across restarts."""

    def __init__(self, cap: int = 2000, path: Path | str | None = None) -> None:
        """Load persisted sessions/aliases, or run memory-only when path is None."""
        self._states: dict[str, SessionState] = {}
        self._aliases: dict[str, str] = {}
        self._responses: dict[str, dict[str, Any]] = {}
        self._guard = asyncio.Lock()
        self._cap = cap
        self._db: sqlite3.Connection | None = None
        if path is not None:
            self._db = connect(path)
            self._load()

    def _evict_session(self) -> None:
        """Drop the oldest session when over cap (caller holds the guard)."""
        if len(self._states) >= self._cap:
            old = next(iter(self._states))
            self._states.pop(old)
            if self._db is not None:
                self._db.execute("DELETE FROM sessions WHERE key=?", (old,))

    def _evict_alias(self) -> None:
        """Drop the oldest alias when over cap (caller holds the guard)."""
        if len(self._aliases) > self._cap * 2:
            old = next(iter(self._aliases))
            self._aliases.pop(old)
            if self._db is not None:
                self._db.execute("DELETE FROM aliases WHERE alias=?", (old,))

    def _evict_response(self) -> None:
        """Drop the oldest cached response when over cap (caller holds guard)."""
        if len(self._responses) >= self._cap:
            old = next(iter(self._responses))
            self._responses.pop(old)
            if self._db is not None:
                self._db.execute("DELETE FROM responses WHERE id=?", (old,))

    def _load(self) -> None:
        """Preload sessions and aliases; response bodies stay lazy (large)."""
        db = self._db
        if db is None:
            return
        rows = db.execute(
            "SELECT key, account, metadata, norm FROM sessions"
            " ORDER BY updated_at ASC, key ASC",
        ).fetchall()
        for key, account, metadata, norm in rows:
            state = SessionState()
            state.account = account
            state.metadata = _loads_list(str(metadata))
            state.norm = [str(v) for v in _loads_list(str(norm))]
            self._states[key] = state
        pairs = db.execute(
            "SELECT alias, target FROM aliases ORDER BY updated_at ASC, alias ASC",
        ).fetchall()
        for alias, target in pairs:
            self._aliases[str(alias)] = str(target)

    def _resolve(self, key: str) -> str:
        for _ in range(5):
            if key not in self._aliases:
                break
            key = self._aliases[key]
        return key

    def close(self) -> None:
        """Flush and close the SQLite handle (no-op when memory-only)."""
        if self._db is not None:
            try:
                self._db.commit()
            finally:
                self._db.close()
            self._db = None

    async def get(self, key: str) -> SessionState | None:
        """Return the session for key, following aliases."""
        async with self._guard:
            return self._states.get(self._resolve(key))

    async def get_or_new(self, key: str) -> tuple[SessionState, bool]:
        """Return the session for key, creating and persisting an empty one."""
        async with self._guard:
            key = self._resolve(key)
            state = self._states.get(key)
            if state is None:
                self._evict_session()
                state = SessionState()
                self._states[key] = state
                if self._db is not None:
                    self._db.execute(
                        "INSERT INTO sessions(key, account, metadata, norm, updated_at)"
                        " VALUES(?, NULL, '[]', '[]', ?)",
                        (key, int(time.time())),
                    )
                    self._db.commit()
                return state, True
            return state, False

    async def persist(self, key: str) -> None:
        """Write the in-memory session for key through to SQLite."""
        async with self._guard:
            if self._db is None:
                return
            key = self._resolve(key)
            state = self._states.get(key)
            if state is None:
                return
            self._db.execute(
                "INSERT INTO sessions(key, account, metadata, norm, updated_at)"
                " VALUES(?, ?, ?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET account=excluded.account,"
                " metadata=excluded.metadata, norm=excluded.norm,"
                " updated_at=excluded.updated_at",
                (
                    key,
                    state.account,
                    _dumps(state.metadata),
                    _dumps(state.norm),
                    int(time.time()),
                ),
            )
            self._db.commit()

    async def lock_for(self, key: str) -> asyncio.Lock:
        """Return the per-conversation lock serializing read-modify-write."""
        state, _ = await self.get_or_new(key)
        return state.lock

    async def link(self, alias: str, key: str) -> None:
        """Alias one key to another, persisted so chains survive restart."""
        async with self._guard:
            target = self._resolve(key)
            self._aliases[alias] = target
            self._evict_alias()
            if self._db is not None:
                self._db.execute(
                    "INSERT INTO aliases(alias, target, updated_at) VALUES(?, ?, ?)"
                    " ON CONFLICT(alias) DO UPDATE SET target=excluded.target,"
                    " updated_at=excluded.updated_at",
                    (alias, target, int(time.time())),
                )
                self._db.commit()

    async def set_norm(self, key: str, norm: list[str]) -> None:
        """Record the input norm covered by server-side history (persisted)."""
        async with self._guard:
            resolved = self._resolve(key)
            state = self._states.get(resolved)
            if state is not None:
                state.norm = list(norm)
                if self._db is not None:
                    self._db.execute(
                        "UPDATE sessions SET norm=?, updated_at=? WHERE key=?",
                        (_dumps(state.norm), int(time.time()), resolved),
                    )
                    self._db.commit()

    async def get_norm(self, key: str) -> list[str] | None:
        """Return the stored input norm for key, or None when unknown."""
        async with self._guard:
            state = self._states.get(self._resolve(key))
            return list(state.norm) if state else None

    async def touch(self, key: str) -> SessionState | None:
        """Move state to MRU end so eviction drops idle sessions first."""
        async with self._guard:
            key = self._resolve(key)
            state = self._states.pop(key, None)
            if state is None:
                return None
            self._states[key] = state
            if self._db is not None:
                self._db.execute(
                    "UPDATE sessions SET updated_at=? WHERE key=?",
                    (int(time.time()), key),
                )
                self._db.commit()
            return state

    async def save_response(self, rid: str, obj: dict[str, Any]) -> None:
        """Save a response object by id, persisted for later chaining/lookup."""
        async with self._guard:
            self._evict_response()
            self._responses[rid] = obj
            if self._db is not None:
                self._db.execute(
                    "INSERT INTO responses(id, body, updated_at) VALUES(?, ?, ?)"
                    " ON CONFLICT(id) DO UPDATE SET body=excluded.body,"
                    " updated_at=excluded.updated_at",
                    (rid, _dumps(obj), int(time.time())),
                )
                total_row = self._db.execute(
                    "SELECT COUNT(*) FROM responses",
                ).fetchone()
                total = int(total_row[0]) if total_row else 0
                if total > self._cap:
                    rows = self._db.execute(
                        "SELECT id FROM responses ORDER BY ROWID ASC LIMIT ?",
                        (total - self._cap,),
                    ).fetchall()
                    evicted = [str(r[0]) for r in rows]
                    self._db.execute(
                        "DELETE FROM responses WHERE id IN (SELECT id FROM responses"
                        " ORDER BY ROWID ASC LIMIT ?)",
                        (len(evicted),),
                    )
                    for eid in evicted:
                        self._responses.pop(eid, None)
                self._db.commit()

    async def get_response(self, rid: str) -> dict[str, Any] | None:
        """Fetch a saved response, falling back to SQLite after a restart."""
        async with self._guard:
            obj = self._responses.get(rid)
            if obj is not None:
                return obj
            if self._db is None:
                return None
            row = self._db.execute(
                "SELECT body FROM responses WHERE id=?",
                (rid,),
            ).fetchone()
            if row is None:
                return None
            try:
                body = json.loads(row[0])
            except ValueError:
                return None
            if not isinstance(body, dict):
                return None
            if len(self._responses) >= self._cap:
                self._responses.pop(next(iter(self._responses)))
            self._responses[rid] = body
            return body
