# Copyright 2026 gem2oai contributors.
"""Session store: conversation keys/aliases -> account-bound Gemini continuation.

Continuation is stored as plain Gemini metadata ([cid, rid, rcid]), so any
request can resume native server-side history with a fresh ChatSession on the
same account. Keys come from Response conversation ids, previous_response_id
chains, or content fingerprints of Chat Completions prefixes.

State is written through to SQLite (config.DB_PATH) on every mutation and
reloaded on boot, so restarts lose no conversations, aliases, or responses.
When Redis is attached (attach_redis), mutations also dual-write to Redis so
a second process or a restart sees the same rows; reads hydrate from Redis
on a local miss and refresh when the shared copy is newer
(last-writer-wins). Redis failures degrade to SQLite per call and never fail
a request. Tier caps are enforced independently: local eviction drops the
memory/SQLite row, Redis trims drop the shared row; SQLite stays the full
archive for boot.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from redis.exceptions import RedisError

from redis_store import RedisState
from store import connect

if TYPE_CHECKING:
    import sqlite3

_log = logging.getLogger("gem2oai")

_REDIS_ERRORS = (RedisError, OSError, TimeoutError)


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

    __slots__ = ("account", "generation", "lock", "metadata", "norm", "updated_at")

    def __init__(self) -> None:
        """Create an empty continuation state with its own lock."""
        self.account: int | None = None
        self.metadata: list[str | None] = []
        self.norm: list[str] = []
        self.updated_at: int = 0
        self.generation: int = 0
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
        self._redis: RedisState | None = None
        self._gen: dict[str, int] = {}
        if path is not None:
            self._db = connect(path)
            self._load()

    def attach_redis(self, state: RedisState | None) -> None:
        """Dual-write/hydrate session state through Redis (None detaches)."""
        self._redis = state

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
            "SELECT key, account, metadata, norm, updated_at FROM sessions"
            " ORDER BY updated_at ASC, key ASC",
        ).fetchall()
        for key, account, metadata, norm, updated_at in rows:
            state = SessionState()
            state.account = account
            state.metadata = _loads_list(str(metadata))
            state.norm = [str(v) for v in _loads_list(str(norm))]
            state.updated_at = int(updated_at or 0)
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

    async def _resolve_async(self, key: str) -> str:
        """Follow aliases through memory, hydrating missing hops from Redis."""
        for _ in range(10):
            async with self._guard:
                nxt = self._aliases.get(key)
            if nxt is not None:
                key = nxt
                continue
            redis = self._redis
            if redis is None:
                break
            try:
                nxt = await redis.alias_get(key)
            except _REDIS_ERRORS as exc:
                _log.warning("gem2oai: Redis alias fetch skipped: %s", exc)
                break
            if nxt is None:
                break
            async with self._guard:
                self._aliases[key] = nxt
                self._evict_alias()
            key = nxt
        return key

    def _upsert_locked(
        self,
        key: str,
        account: int | None,
        metadata: list[str | None],
        norm: list[str],
        updated_at: int,
    ) -> None:
        """Write one session row to SQLite (caller holds the guard)."""
        if self._db is None:
            return
        self._db.execute(
            "INSERT INTO sessions(key, account, metadata, norm, updated_at)"
            " VALUES(?, ?, ?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET account=excluded.account,"
            " metadata=excluded.metadata, norm=excluded.norm,"
            " updated_at=excluded.updated_at",
            (key, account, _dumps(metadata), _dumps(norm), updated_at),
        )
        self._db.commit()

    async def _push_session(self, key: str, *, fresh: bool = False) -> None:
        """Best-effort dual-write of one memory session to Redis (off guard)."""
        redis = self._redis
        if redis is None:
            return
        async with self._guard:
            state = self._states.get(key)
            if state is None:
                return
            fresh_snap = (
                state.account,
                list(state.metadata),
                list(state.norm),
                state.updated_at or int(time.time()),
            )
        try:
            if fresh:
                # Never clobber a real row with a just-created empty one:
                # only write when the shared key is still absent.
                existing = await redis.session_get(key)
                if existing is not None and (
                    existing.get("metadata") or existing.get("norm")
                ):
                    async with self._guard:
                        current = self._states.get(key)
                        if current is not None and not (
                            current.metadata or current.norm
                        ):
                            self._states.pop(key, None)
                    await self._sync_from_redis(key, None)
                    return
            await redis.session_put(key, *fresh_snap)
            for evicted in await redis.trim_sessions(self._cap):
                async with self._guard:
                    self._states.pop(evicted, None)
        except _REDIS_ERRORS as exc:
            _log.warning("gem2oai: Redis session write skipped: %s", exc)

    async def _sync_from_redis(
        self,
        key: str,
        state: SessionState | None,
    ) -> SessionState | None:
        """Hydrate (miss) or refresh (stale) one session from Redis.

        Returns the memory state, or None when Redis is absent/unreachable.
        A bumped delete-generation tombstones stale memory rows so a drop in
        another process stays dropped.
        """
        redis = self._redis
        if redis is None:
            return state
        try:
            gen, data = await redis.session_get_with_generation(key)
        except _REDIS_ERRORS as exc:
            _log.warning("gem2oai: Redis session fetch skipped: %s", exc)
            return state
        async with self._guard:
            known = self._gen.get(key, 0)
            if gen > known:
                self._gen[key] = gen
                if data is None:
                    self._states.pop(key, None)
                    return None
                current = self._states.get(key)
                if current is None:
                    self._evict_session()
                    current = SessionState()
                    current.generation = gen
                    self._states[key] = current
            else:
                current = self._states.get(key)
                if data is None:
                    return current
                if current is None:
                    self._evict_session()
                    current = SessionState()
                    current.generation = self._gen.get(key, 0)
                    self._states[key] = current
            incoming = data.get("updated_at", 0)
            incoming_at = incoming if isinstance(incoming, int) else 0
            remote_norm = data.get("norm", [])
            remote_norm_list = (
                [str(v) for v in remote_norm] if isinstance(remote_norm, list) else []
            )
            remote_meta = data.get("metadata", [])
            remote_meta_list = (
                list(remote_meta) if isinstance(remote_meta, list) else []
            )
            if incoming_at >= current.updated_at and (
                data.get("account") != current.account
                or remote_meta_list != list(current.metadata)
                or remote_norm_list != list(current.norm)
            ):
                account = data.get("account")
                current.account = account if isinstance(account, int) else None
                current.metadata = remote_meta_list
                current.norm = remote_norm_list
                current.updated_at = incoming_at
                self._upsert_locked(
                    key,
                    current.account,
                    current.metadata,
                    current.norm,
                    incoming_at,
                )
            return current

    def close(self) -> None:
        """Flush and close the SQLite handle (no-op when memory-only)."""
        if self._db is not None:
            try:
                self._db.commit()
            finally:
                self._db.close()
            self._db = None

    async def get(self, key: str) -> SessionState | None:
        """Return the session for key, following aliases (Redis-refreshed)."""
        resolved = await self._resolve_async(key)
        async with self._guard:
            state = self._states.get(resolved)
        return await self._sync_from_redis(resolved, state)

    async def get_or_new(self, key: str) -> tuple[SessionState, bool]:
        """Return the session for key, creating and persisting an empty one."""
        resolved = await self._resolve_async(key)
        async with self._guard:
            state = self._states.get(resolved)
            if state is not None:
                return state, False
        hydrated = await self._sync_from_redis(resolved, None)
        if hydrated is not None:
            return hydrated, False
        async with self._guard:
            state = self._states.get(resolved)
            if state is not None:
                return state, False
            self._evict_session()
            state = SessionState()
            state.updated_at = int(time.time())
            self._states[resolved] = state
            if self._db is not None:
                try:
                    self._db.execute(
                        "INSERT INTO sessions(key, account, metadata, norm, updated_at)"
                        " VALUES(?, NULL, '[]', '[]', ?)",
                        (resolved, state.updated_at),
                    )
                    self._db.commit()
                except Exception as exc:  # noqa: BLE001 - concurrent create loses
                    _log.debug("gem2oai: session create raced, reusing row: %s", exc)
                    row = self._db.execute(
                        "SELECT account, metadata, norm, updated_at FROM sessions"
                        " WHERE key=?",
                        (resolved,),
                    ).fetchone()
                    if row is not None:
                        self._states.pop(resolved, None)
                        state = SessionState()
                        state.account = row[0]
                        state.metadata = _loads_list(str(row[1]))
                        state.norm = [str(v) for v in _loads_list(str(row[2]))]
                        state.updated_at = int(row[3] or 0)
                        self._states[resolved] = state
                        return state, False
        await self._push_session(resolved, fresh=True)
        return state, True

    async def _drop_redis(self, key: str, resolved: str) -> None:
        """Best-effort removal of one session and its aliases from Redis."""
        redis = self._redis
        if redis is None:
            return
        try:
            await redis.session_delete(resolved)
            await redis.alias_remove_for(resolved)
            if key != resolved:
                await redis.alias_remove_for(key)
            gen, _ = await redis.session_get_with_generation(resolved)
        except _REDIS_ERRORS as exc:
            _log.warning("gem2oai: Redis session delete skipped: %s", exc)
            return
        async with self._guard:
            self._gen[resolved] = gen

    async def drop(self, key: str) -> None:
        """Forget one session row (memory + SQLite + Redis) so it restarts fresh."""
        resolved = await self._resolve_async(key)
        async with self._guard:
            self._states.pop(resolved, None)
            self._aliases = {
                a: t for a, t in self._aliases.items() if a != key and t != resolved
            }
            if self._db is not None:
                self._db.execute("DELETE FROM sessions WHERE key=?", (resolved,))
                self._db.execute(
                    "DELETE FROM aliases WHERE alias=? OR target=?",
                    (key, resolved),
                )
                self._db.commit()
        await self._drop_redis(key, resolved)

    async def drop_if_fresh(self, key: str) -> bool:
        """Drop key only when it holds no continuation metadata and no norm.

        Fresh keys gain an account via _pick_account before the first turn;
        when that turn fails nothing resumable exists, so the row is junk
        that only pressures the cap. Keys with history are never dropped.
        Unknown keys (nowhere) are never dropped.
        """
        resolved = await self._resolve_async(key)
        async with self._guard:
            state = self._states.get(resolved)
            if state is not None and (state.metadata or state.norm):
                return False
            seen = state is not None
        synced = await self._sync_from_redis(resolved, state)
        if synced is not None:
            seen = True
            if synced.metadata or synced.norm:
                return False
        if not seen:
            return False
        async with self._guard:
            self._states.pop(resolved, None)
            self._aliases = {
                a: t for a, t in self._aliases.items() if a != key and t != resolved
            }
            if self._db is not None:
                self._db.execute("DELETE FROM sessions WHERE key=?", (resolved,))
                self._db.execute(
                    "DELETE FROM aliases WHERE alias=? OR target=?",
                    (key, resolved),
                )
                self._db.commit()
        await self._drop_redis(key, resolved)
        return True

    async def drop_if(
        self,
        key: str,
        account: int | None,
        metadata: list[str | None],
    ) -> bool:
        """Drop key only if it still holds the snapshotted account/metadata.

        Unknown keys (no row in any tier) return True without touching state:
        _purge_dead_continuations snapshots call this, and Redis-only rows
        check both tiers before removing. Redis is checked first via an atomic
        Lua compare-and-delete so a concurrent update cannot slip between the
        check and the delete; local state only drops when Redis agrees.
        """
        resolved = await self._resolve_async(key)
        redis = self._redis
        if redis is not None:
            try:
                account_raw = "" if account is None else str(account)
                kept = await redis.session_drop_if(
                    resolved,
                    account_raw,
                    _dumps(metadata),
                )
            except _REDIS_ERRORS as exc:
                _log.warning("gem2oai: Redis session fetch skipped: %s", exc)
                async with self._guard:
                    return self._states.get(resolved) is not None
            if not kept:
                return False
            gen, _ = await redis.session_get_with_generation(resolved)
            async with self._guard:
                self._gen[resolved] = gen
        async with self._guard:
            state = self._states.get(resolved)
            if state is not None and (
                state.account != account or list(state.metadata) != list(metadata)
            ):
                return False
            if state is None:
                return redis is None
            self._states.pop(resolved, None)
            self._aliases = {
                a: t for a, t in self._aliases.items() if a != key and t != resolved
            }
            if self._db is not None:
                self._db.execute("DELETE FROM sessions WHERE key=?", (resolved,))
                self._db.execute(
                    "DELETE FROM aliases WHERE alias=? OR target=?",
                    (key, resolved),
                )
                self._db.commit()
        if redis is not None:
            await self._drop_redis(key, resolved)
        return True

    async def persist(self, key: str) -> None:
        """Write the in-memory session for key through to SQLite and Redis."""
        resolved = await self._resolve_async(key)
        async with self._guard:
            state = self._states.get(resolved)
            if state is None:
                return
            state.updated_at = int(time.time())
            self._upsert_locked(
                resolved,
                state.account,
                state.metadata,
                state.norm,
                state.updated_at,
            )
        await self._push_session(resolved)

    def snapshot(self) -> list[tuple[str, int | None, list[str | None]]]:
        """Return (key, account, metadata) for every memory session row."""
        return [(k, s.account, list(s.metadata)) for k, s in self._states.items()]

    async def snapshot_all(self) -> list[tuple[str, int | None, list[str | None]]]:
        """Return memory rows plus Redis-only rows (for cross-process purge)."""
        async with self._guard:
            rows = [(k, s.account, list(s.metadata)) for k, s in self._states.items()]
        redis = self._redis
        if redis is None:
            return rows
        try:
            remote = await redis.session_snapshot(self._cap)
        except _REDIS_ERRORS as exc:
            _log.warning("gem2oai: Redis snapshot skipped: %s", exc)
            return rows
        seen = {k for k, _, _ in rows}
        rows.extend(
            (key, account, list(meta))
            for key, account, meta in remote
            if key not in seen
        )
        return rows

    async def lock_for(self, key: str) -> asyncio.Lock:
        """Return the per-conversation lock serializing read-modify-write."""
        state, _ = await self.get_or_new(key)
        return state.lock

    async def link(self, alias: str, key: str) -> None:
        """Alias one key to another, persisted so chains survive restart."""
        target = await self._resolve_async(key)
        async with self._guard:
            target = self._resolve(target)
            self._aliases[alias] = target
            self._evict_alias()
            now = int(time.time())
            if self._db is not None:
                self._db.execute(
                    "INSERT INTO aliases(alias, target, updated_at) VALUES(?, ?, ?)"
                    " ON CONFLICT(alias) DO UPDATE SET target=excluded.target,"
                    " updated_at=excluded.updated_at",
                    (alias, target, now),
                )
                self._db.commit()
        redis = self._redis
        if redis is not None:
            try:
                await redis.alias_put(alias, target, now)
                await redis.trim_aliases(self._cap * 2)
            except _REDIS_ERRORS as exc:
                _log.warning("gem2oai: Redis alias write skipped: %s", exc)

    async def set_norm(self, key: str, norm: list[str]) -> None:
        """Record the input norm covered by server-side history (persisted)."""
        resolved = await self._resolve_async(key)
        async with self._guard:
            state = self._states.get(resolved)
            if state is None:
                return
            state.norm = list(norm)
            state.updated_at = int(time.time())
            if self._db is not None:
                self._db.execute(
                    "UPDATE sessions SET norm=?, updated_at=? WHERE key=?",
                    (_dumps(state.norm), state.updated_at, resolved),
                )
                self._db.commit()
        await self._push_session(resolved)

    async def get_norm(self, key: str) -> list[str] | None:
        """Return the stored input norm for key, or None when unknown."""
        state = await self.get(key)
        return list(state.norm) if state else None

    async def touch(self, key: str) -> SessionState | None:
        """Move state to MRU end so eviction drops idle sessions first."""
        resolved = await self._resolve_async(key)
        async with self._guard:
            state = self._states.pop(resolved, None)
            if state is None:
                return None
            self._states[resolved] = state
            state.updated_at = int(time.time())
            if self._db is not None:
                self._db.execute(
                    "UPDATE sessions SET updated_at=? WHERE key=?",
                    (state.updated_at, resolved),
                )
                self._db.commit()
        redis = self._redis
        if redis is not None:
            try:
                await redis.session_touch(resolved, state.updated_at)
            except _REDIS_ERRORS as exc:
                _log.warning("gem2oai: Redis touch skipped: %s", exc)
        return state

    def _remember_response(self, rid: str, body: dict[str, Any]) -> None:
        """Cache one response body, evicting the oldest (caller holds guard)."""
        if len(self._responses) >= self._cap:
            self._responses.pop(next(iter(self._responses)))
        self._responses[rid] = body

    def _load_response_sqlite_locked(self, rid: str) -> dict[str, Any] | None:
        """Read one response body from SQLite (caller holds the guard)."""
        if self._db is None:
            return None
        row = self._db.execute(
            "SELECT body FROM responses WHERE id=?",
            (rid,),
        ).fetchone()
        if row is None:
            return None
        try:
            parsed = json.loads(row[0])
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    async def _fetch_response_redis(self, rid: str) -> dict[str, Any] | None:
        """Fetch one response body from Redis, caching it locally."""
        redis = self._redis
        if redis is None:
            async with self._guard:
                return self._responses.get(rid)
        try:
            gen, remote = await redis.response_get_with_generation(rid)
        except _REDIS_ERRORS as exc:
            _log.warning("gem2oai: Redis response fetch skipped: %s", exc)
            async with self._guard:
                return self._responses.get(rid)
        async with self._guard:
            known = self._gen.get(f"resp:{rid}", 0)
            if gen > known:
                self._gen[f"resp:{rid}"] = gen
                self._responses.pop(rid, None)
                if remote is None:
                    return None
            if remote is None:
                return self._responses.get(rid)
            self._remember_response(rid, remote)
        return remote

    async def save_response(self, rid: str, obj: dict[str, Any]) -> None:
        """Save a response object by id, persisted for later chaining/lookup."""
        async with self._guard:
            self._evict_response()
            self._responses[rid] = obj
            now = int(time.time())
            if self._db is not None:
                self._db.execute(
                    "INSERT INTO responses(id, body, updated_at) VALUES(?, ?, ?)"
                    " ON CONFLICT(id) DO UPDATE SET body=excluded.body,"
                    " updated_at=excluded.updated_at",
                    (rid, _dumps(obj), now),
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
        redis = self._redis
        if redis is not None:
            try:
                await redis.response_put(rid, obj, now)
                for evicted in await redis.trim_responses(self._cap):
                    async with self._guard:
                        self._responses.pop(evicted, None)
            except _REDIS_ERRORS as exc:
                _log.warning("gem2oai: Redis response write skipped: %s", exc)

    async def get_response(self, rid: str) -> dict[str, Any] | None:
        """Fetch a saved response, falling back to SQLite/Redis after restart."""
        async with self._guard:
            obj = self._responses.get(rid)
            body = self._load_response_sqlite_locked(rid)
            if body is not None:
                self._remember_response(rid, body)
                return body
            if obj is not None and self._redis is None:
                return obj
        return await self._fetch_response_redis(rid)
