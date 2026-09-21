# Copyright 2026 gem2oai contributors.
"""Redis shared state: sessions, aliases, responses, files.

Dual-write with SQLite (store.py): Redis is the cross-process layer so two
checkouts or a restart see the same conversations and /v1/files ids; SQLite
remains the durable fallback when Redis is down or unconfigured.

Key layout (redis-core: lowercase colon-separated):
  gem2oai:session:<key>    Hash {account, metadata, norm, updated_at}
  gem2oai:alias:<alias>     String target (+ gem2oai:alias-rev:<target> Set)
  gem2oai:response:<rid>    String JSON body
  gem2oai:file:<fid>        Hash {data, mime, filename, purpose, created}
  gem2oai:sessions|aliases|responses|files  ZSet member -> updated/created score

ZSets (not KEYS/SCAN) drive eviction and listing; bulk reads use one
non-transactional pipeline (redis-connections). No hash tags: standalone
single-master Valkey; re-plan tags if this ever moves to a cluster
(redis-clustering). No client-side caching: session/file state is
write-heavy, invalidation would overrun the savings.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import redis.asyncio as redis_async
from redis.exceptions import RedisError

_log = logging.getLogger("gem2oai")

_PREFIX = "gem2oai:"
_SESSIONS_IDX = _PREFIX + "sessions"
_ALIASES_IDX = _PREFIX + "aliases"
_RESPONSES_IDX = _PREFIX + "responses"
_FILES_IDX = _PREFIX + "files"

# Lua: repoint alias atomically (server-side read of old target keeps the
# reverse index exact under cross-process races; plain GET+pipeline can leave
# the alias in two rev sets). KEYS: alias key, alias index, new rev set.
# The old rev set is derived server-side from the previous value.
_ALIAS_CAS = """
local cur = redis.call('GET', KEYS[1])
if cur and cur ~= ARGV[1] then
  redis.call('SREM', 'gem2oai:alias-rev:' .. cur, ARGV[3])
end
redis.call('SET', KEYS[1], ARGV[1])
redis.call('ZADD', KEYS[2], ARGV[2], ARGV[3])
redis.call('SADD', KEYS[3], ARGV[3])
return cur
"""

# Lua: drop one session only when account+metadata still match; returns 1 on
# delete, 0 when the row changed underneath (avoids check-then-delete races).
_SESSION_DROP_IF = """
local account = redis.call('HGET', KEYS[1], 'account')
local metadata = redis.call('HGET', KEYS[1], 'metadata')
if account == false then
  return 1
end
if (account or '') ~= ARGV[1] or (metadata or '') ~= ARGV[2] then
  return 0
end
redis.call('DEL', KEYS[1])
redis.call('ZREM', KEYS[2], ARGV[3])
redis.call('INCR', KEYS[3])
return 1
"""


def _skey(key: str) -> str:
    """Return the session Hash key for a conversation key."""
    return _PREFIX + "session:" + key


def _akey(alias: str) -> str:
    """Return the alias String key for an alias."""
    return _PREFIX + "alias:" + alias


def _revkey(target: str) -> str:
    """Return the reverse-index Set key for an alias target."""
    return _PREFIX + "alias-rev:" + target


def _rkey(rid: str) -> str:
    """Return the response String key for a response id."""
    return _PREFIX + "response:" + rid


def _fkey(fid: str) -> str:
    """Return the file Hash key for a file id."""
    return _PREFIX + "file:" + fid


def _dumps(obj: object) -> str:
    """Encode state for Redis, stringifying exotic values."""
    return json.dumps(obj, default=str)


def _safe_int(raw: str | None) -> int:
    """Parse a Redis integer field, returning 0 for corrupt values."""
    text = (raw or "").strip()
    if not text:
        return 0
    if text[0] in ("+", "-"):
        return int(text[1:]) if text[1:].isdigit() else 0
    return int(text) if text.isdigit() else 0


class RedisState:
    """Async Redis handle for shared conversation/file state.

    None means Redis is disabled (unreachable at boot): callers fall back
    to SQLite. Every method may raise RedisError/OSError on a live handle;
    stores catch those and degrade to SQLite per call.
    """

    def __init__(self, client: redis_async.Redis) -> None:
        """Wrap an already-connected async Redis client."""
        self._client = client

    @classmethod
    async def create(cls, url: str) -> RedisState | None:
        """Connect with a pooled client; None (SQLite-only) when disabled/down."""
        if not url.strip():
            return None
        pool = redis_async.ConnectionPool.from_url(
            url,
            max_connections=50,
            socket_connect_timeout=2.0,
            socket_timeout=5.0,
            retry_on_timeout=True,
        )
        client = redis_async.Redis(connection_pool=pool)
        try:
            await client.ping()
        except (RedisError, OSError, TimeoutError) as exc:
            _log.warning("gem2oai: Redis at %s unreachable, SQLite-only: %s", url, exc)
            await client.aclose()
            return None
        return cls(client)

    async def aclose(self) -> None:
        """Close the pool; never raises into lifespan shutdown."""
        try:
            await self._client.aclose()
        except (RedisError, OSError, TimeoutError) as exc:
            _log.warning("gem2oai: Redis close skipped: %s", exc)

    async def ping_ms(self) -> float:
        """Return the PING round-trip in milliseconds."""
        start = time.monotonic()
        await self._client.ping()
        return (time.monotonic() - start) * 1000.0

    async def stats(self) -> dict[str, Any]:
        """Return hit-ratio/throughput/memory signals for /health."""
        info = await self._client.info("stats")
        mem = await self._client.info("memory")
        hits = int(info.get("keyspace_hits", 0))
        misses = int(info.get("keyspace_misses", 0))
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_ratio": (hits / total) if total else None,
            "ops_per_sec": int(info.get("instantaneous_ops_per_sec", 0)),
            "connected_clients": int(info.get("connected_clients", 0)),
            "used_memory": str(mem.get("used_memory_human", "?")),
        }

    async def _trim(self, index: str, prefix: str, cap: int) -> list[str]:
        """Drop oldest index members beyond cap; return evicted member names."""
        total = await self._client.zcard(index)
        if total <= cap:
            return []
        overflow = total - cap
        members = await self._client.zrange(index, 0, overflow - 1)
        evicted = [m.decode() if isinstance(m, bytes) else str(m) for m in members]
        if not evicted:
            return []
        pipe = self._client.pipeline(transaction=False)
        pipe.zrem(index, *evicted)
        for member in evicted:
            pipe.delete(prefix + member)
        await pipe.execute()
        return evicted

    async def session_get(self, key: str) -> dict[str, Any] | None:
        """Fetch one session Hash, or None when absent."""
        raw = await self._client.hgetall(_skey(key))
        if not raw:
            return None

        def field(name: str) -> str:
            val = raw.get(name.encode(), b"")
            return val.decode() if isinstance(val, bytes) else str(val)

        account_raw = field("account")
        try:
            metadata = json.loads(field("metadata") or "[]")
        except ValueError:
            metadata = []
        try:
            norm = json.loads(field("norm") or "[]")
        except ValueError:
            norm = []
        return {
            "account": int(account_raw) if account_raw.isdigit() else None,
            "metadata": list(metadata) if isinstance(metadata, list) else [],
            "norm": [str(v) for v in norm] if isinstance(norm, list) else [],
            "updated_at": _safe_int(field("updated_at")),
        }

    async def session_put(
        self,
        key: str,
        account: int | None,
        metadata: object,
        norm: object,
        updated_at: int,
    ) -> None:
        """Write one session Hash plus its recency index entry in one trip."""
        pipe = self._client.pipeline(transaction=False)
        pipe.hset(
            _skey(key),
            mapping={
                "account": "" if account is None else str(account),
                "metadata": _dumps(metadata),
                "norm": _dumps(norm),
                "updated_at": str(updated_at),
            },
        )
        pipe.zadd(_SESSIONS_IDX, {key: updated_at})
        await pipe.execute()

    async def session_delete(self, key: str) -> None:
        """Delete one session Hash, leaving a tombstone generation behind."""
        pipe = self._client.pipeline(transaction=False)
        pipe.delete(_skey(key))
        pipe.zrem(_SESSIONS_IDX, key)
        pipe.incr(_PREFIX + "session-gen:" + key)
        await pipe.execute()

    async def session_drop_if(self, key: str, account: str, metadata: str) -> bool:
        """Delete a session only when account+metadata still match (atomic)."""
        dropped = await self._client.eval(
            _SESSION_DROP_IF,
            3,
            _skey(key),
            _SESSIONS_IDX,
            _PREFIX + "session-gen:" + key,
            account,
            metadata,
            key,
        )
        return bool(dropped)

    async def session_get_with_generation(
        self,
        key: str,
    ) -> tuple[int, dict[str, Any] | None]:
        """Fetch one session plus its delete-generation in a single trip."""
        pipe = self._client.pipeline(transaction=False)
        pipe.get(_PREFIX + "session-gen:" + key)
        pipe.hgetall(_skey(key))
        gen_raw, raw = await pipe.execute()
        gen_text = (
            gen_raw.decode()
            if isinstance(gen_raw, bytes)
            else (str(gen_raw) if gen_raw is not None else "0")
        )
        gen = int(gen_text) if gen_text.isdigit() else 0
        if not raw:
            return gen, None

        def field(name: str) -> str:
            val = raw.get(name.encode(), b"")
            return val.decode() if isinstance(val, bytes) else str(val)

        account_raw = field("account")
        try:
            metadata = json.loads(field("metadata") or "[]")
        except ValueError:
            metadata = []
        try:
            norm = json.loads(field("norm") or "[]")
        except ValueError:
            norm = []
        return gen, {
            "account": int(account_raw) if account_raw.isdigit() else None,
            "metadata": list(metadata) if isinstance(metadata, list) else [],
            "norm": [str(v) for v in norm] if isinstance(norm, list) else [],
            "updated_at": _safe_int(field("updated_at")),
        }

    async def session_generation(self, key: str) -> int:
        """Return the delete-generation counter for a session key."""
        gen, _ = await self.session_get_with_generation(key)
        return gen

    async def session_touch(self, key: str, updated_at: int) -> None:
        """Refresh recency only when the session Hash still exists (no ghosts)."""
        touched = await self._client.eval(
            "if redis.call('EXISTS', KEYS[1]) == 1 then "
            "redis.call('HSET', KEYS[1], 'updated_at', ARGV[1]) "
            "redis.call('ZADD', KEYS[2], ARGV[1], ARGV[2]) return 1 end return 0",
            2,
            _skey(key),
            _SESSIONS_IDX,
            str(updated_at),
            key,
        )
        _ = touched

    async def session_snapshot(self, limit: int) -> list[tuple[str, int | None, list]]:
        """Return (key, account, metadata) for indexed sessions, oldest first."""
        keys = await self._client.zrange(_SESSIONS_IDX, 0, limit - 1)
        if not keys:
            return []
        names = [k.decode() if isinstance(k, bytes) else str(k) for k in keys]
        pipe = self._client.pipeline(transaction=False)
        for name in names:
            pipe.hmget(_skey(name), "account", "metadata")
        rows = await pipe.execute()
        out: list[tuple[str, int | None, list]] = []
        for name, row in zip(names, rows, strict=True):
            if not row or row[0] is None:
                continue
            acct_raw = row[0].decode() if isinstance(row[0], bytes) else str(row[0])
            try:
                meta = json.loads(row[1] or "[]")
            except (ValueError, TypeError):
                meta = []
            out.append(
                (
                    name,
                    int(acct_raw) if acct_raw.isdigit() else None,
                    list(meta) if isinstance(meta, list) else [],
                ),
            )
        return out

    async def trim_sessions(self, cap: int) -> list[str]:
        """Evict oldest sessions beyond cap; return evicted keys."""
        return await self._trim(_SESSIONS_IDX, _PREFIX + "session:", cap)

    async def alias_get(self, alias: str) -> str | None:
        """Return the target for an alias, or None when absent."""
        val = await self._client.get(_akey(alias))
        if val is None:
            return None
        return val.decode() if isinstance(val, bytes) else str(val)

    async def alias_put(self, alias: str, target: str, updated_at: int) -> None:
        """Point an alias at a target, tracking the reverse index atomically."""
        await self._client.eval(
            _ALIAS_CAS,
            3,
            _akey(alias),
            _ALIASES_IDX,
            _revkey(target),
            target,
            updated_at,
            alias,
        )

    async def alias_remove_for(self, alias_or_target: str) -> None:
        """Drop the alias key plus every alias pointing at the target."""
        members = await self._client.smembers(_revkey(alias_or_target))
        names = {alias_or_target}
        names.update(m.decode() if isinstance(m, bytes) else str(m) for m in members)
        pipe = self._client.pipeline(transaction=False)
        for name in names:
            pipe.delete(_akey(name))
            pipe.srem(_revkey(alias_or_target), name)
        pipe.zrem(_ALIASES_IDX, *names)
        pipe.delete(_revkey(alias_or_target))
        await pipe.execute()

    async def trim_aliases(self, cap: int) -> list[str]:
        """Evict oldest aliases beyond cap, pruning rev sets by old target."""
        total = await self._client.zcard(_ALIASES_IDX)
        if total <= cap:
            return []
        overflow = total - cap
        members = await self._client.zrange(_ALIASES_IDX, 0, overflow - 1)
        evicted = [m.decode() if isinstance(m, bytes) else str(m) for m in members]
        if not evicted:
            return []
        pipe = self._client.pipeline(transaction=False)
        for alias in evicted:
            pipe.get(_akey(alias))
        old_targets = await pipe.execute()
        pipe = self._client.pipeline(transaction=False)
        pipe.zrem(_ALIASES_IDX, *evicted)
        for alias, old in zip(evicted, old_targets, strict=True):
            pipe.delete(_akey(alias))
            target = old.decode() if isinstance(old, bytes) else old
            if isinstance(target, str) and target:
                pipe.srem(_revkey(target), alias)
        await pipe.execute()
        return evicted

    async def response_get(self, rid: str) -> dict[str, Any] | None:
        """Fetch one response body, or None when absent/corrupt."""
        gen, body = await self.response_get_with_generation(rid)
        _ = gen
        return body

    async def response_get_with_generation(
        self,
        rid: str,
    ) -> tuple[int, dict[str, Any] | None]:
        """Fetch one response plus its delete-generation in a single trip."""
        pipe = self._client.pipeline(transaction=False)
        pipe.get(_PREFIX + "response-gen:" + rid)
        pipe.get(_rkey(rid))
        gen_raw, raw = await pipe.execute()
        gen_text = (
            gen_raw.decode()
            if isinstance(gen_raw, bytes)
            else (str(gen_raw) if gen_raw is not None else "0")
        )
        gen = int(gen_text) if gen_text.isdigit() else 0
        if raw is None:
            return gen, None
        text = raw.decode() if isinstance(raw, bytes) else str(raw)
        try:
            body = json.loads(text)
        except ValueError:
            return gen, None
        return gen, body if isinstance(body, dict) else None

    async def response_put(self, rid: str, body: object, updated_at: int) -> None:
        """Write one response body plus its recency index entry in one trip."""
        pipe = self._client.pipeline(transaction=False)
        pipe.set(_rkey(rid), _dumps(body))
        pipe.zadd(_RESPONSES_IDX, {rid: updated_at})
        await pipe.execute()

    async def response_delete(self, rid: str) -> None:
        """Delete one response body, leaving a tombstone generation behind."""
        pipe = self._client.pipeline(transaction=False)
        pipe.delete(_rkey(rid))
        pipe.zrem(_RESPONSES_IDX, rid)
        pipe.incr(_PREFIX + "response-gen:" + rid)
        await pipe.execute()

    async def trim_responses(self, cap: int) -> list[str]:
        """Evict oldest responses beyond cap; return evicted ids."""
        return await self._trim(_RESPONSES_IDX, _PREFIX + "response:", cap)

    async def file_get(self, fid: str) -> tuple[bytes, str, str, str, int] | None:
        """Fetch one file Hash, or None when absent."""
        data, mime, filename, purpose, created = await self._client.hmget(
            _fkey(fid),
            "data",
            "mime",
            "filename",
            "purpose",
            "created",
        )
        if data is None:
            return None

        def text(val: object, default: str) -> str:
            if val is None:
                return default
            return val.decode() if isinstance(val, bytes) else str(val)

        raw_created = text(created, "0")
        payload = data if isinstance(data, bytes) else str(data).encode()
        return (
            payload,
            text(mime, "application/octet-stream"),
            text(filename, "upload.bin"),
            text(purpose, "assistants"),
            int(raw_created) if raw_created.isdigit() else 0,
        )

    async def file_put(
        self,
        fid: str,
        record: tuple[bytes, str, str, str, int],
    ) -> None:
        """Write one file Hash plus its index entry in one trip."""
        data, mime, filename, purpose, created = record
        pipe = self._client.pipeline(transaction=False)
        pipe.hset(
            _fkey(fid),
            mapping={
                "data": data,
                "mime": mime,
                "filename": filename,
                "purpose": purpose,
                "created": str(created),
            },
        )
        pipe.zadd(_FILES_IDX, {fid: created})
        await pipe.execute()

    async def file_delete(self, fid: str) -> bool:
        """Delete one file Hash; True when the id existed."""
        pipe = self._client.pipeline(transaction=False)
        pipe.delete(_fkey(fid))
        pipe.zrem(_FILES_IDX, fid)
        results = await pipe.execute()
        return bool(results[0])

    async def file_list(self) -> list[tuple[str, int, str, str, int]]:
        """List (id, size, filename, purpose, created), oldest first.

        Sizes come from HSTRLEN so listing never pulls file bytes.
        """
        members = await self._client.zrange(_FILES_IDX, 0, -1)
        if not members:
            return []
        fids = [m.decode() if isinstance(m, bytes) else str(m) for m in members]
        pipe = self._client.pipeline(transaction=False)
        for fid in fids:
            pipe.hstrlen(_fkey(fid), "data")
            pipe.hmget(_fkey(fid), "filename", "purpose", "created")
        rows = await pipe.execute()
        out: list[tuple[str, int, str, str, int]] = []
        for fid, size, meta in zip(fids, rows[0::2], rows[1::2], strict=True):
            if not isinstance(meta, list) or meta[0] is None:
                continue

            def text(val: object, default: str) -> str:
                if val is None:
                    return default
                return val.decode() if isinstance(val, bytes) else str(val)

            created_raw = text(meta[2], "0")
            out.append(
                (
                    fid,
                    int(size),
                    text(meta[0], "upload.bin"),
                    text(meta[1], "assistants"),
                    int(created_raw) if created_raw.isdigit() else 0,
                ),
            )
        return out

    async def trim_files(self, cap: int) -> list[str]:
        """Evict oldest files beyond cap; return evicted ids."""
        return await self._trim(_FILES_IDX, _PREFIX + "file:", cap)
