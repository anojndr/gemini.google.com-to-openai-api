# Copyright 2026 gem2oai contributors.
"""Account pool: parse Netscape cookie blocks, round-robin GeminiClients.

accounts.txt layout: free text with one Netscape cookie jar per ``` block.
Any number of blocks supported; each block needs __Secure-1PSID.
Live rotation (1PSIDTS refresh, full-jar cookie churn) syncs back into the
file via sync_accounts_file, so restarts pick up the newest values.
"""

from __future__ import annotations

import asyncio
import itertools
import re
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gemini_webapi import GeminiClient
from gemini_webapi.constants import AccountStatus

_BLOCK_RE = re.compile(r"```\n(.*?)```", re.DOTALL)
_JAR_COLUMNS = 7
_COOLDOWN_STRIKES = 3


def _parse_cookie_block(block: str) -> dict[str, str]:
    """Parse one Netscape cookie-jar block into a name -> value mapping."""
    cookies: dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#HttpOnly"):
            line = line[len("#HttpOnly") :].lstrip()
        elif line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= _JAR_COLUMNS:
            cookies[parts[5]] = parts[6]
    return cookies


def load_account_cookies(path: str) -> list[dict[str, str]]:
    """Extract cookie dicts from every ``` jar block; skip blocks without auth."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = _BLOCK_RE.findall(raw)
    accounts = [_parse_cookie_block(b) for b in blocks]
    return [c for c in accounts if c.get("__Secure-1PSID")]


_JAR_DOMAIN = ".google.com"


def _pool_jars(pool: AccountPool) -> list[dict[str, Any]]:
    """Snapshot live cookie objects per account, keyed by cookie name."""
    jars: list[dict[str, Any]] = []
    for index in range(pool.client_count()):
        try:
            jar = pool.client_at(index).cookies
        except (AttributeError, TypeError, ValueError):
            continue
        try:
            cookies = {str(c.name): c for c in jar.jar if c.value}
        except (AttributeError, TypeError, ValueError):
            continue
        psid = cookies.get("__Secure-1PSID")
        if psid is None or not psid.value:
            continue
        jars.append(cookies)
    return jars


def pool_live_state(
    pool: AccountPool,
) -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, dict[str, object]]],
]:
    """Snapshot live values, expiries, and jar attributes in one pass.

    Returns (live, expiries, attrs): value dicts per account, expiry
    columns keyed by __Secure-1PSID then cookie name, and domain/path/
    secure/expiry attributes in the same keying. One pass, so values
    and expiries cannot skew mid-rotation. Best-effort: an unreadable
    jar is dropped, never raised.
    """
    live: list[dict[str, str]] = []
    expiries: dict[str, dict[str, str]] = {}
    attrs: dict[str, dict[str, dict[str, object]]] = {}
    for jar in _pool_jars(pool):
        psid = str(jar["__Secure-1PSID"].value)
        live.append({name: str(cookie.value) for name, cookie in jar.items()})
        per_exp: dict[str, str] = {}
        per_attr: dict[str, dict[str, object]] = {}
        for name, cookie in jar.items():
            try:
                expires = cookie.expires
            except (AttributeError, TypeError, ValueError):
                expires = None
            if expires:
                per_exp[name] = str(expires)
            try:
                domain = str(cookie.domain or _JAR_DOMAIN)
                path = str(cookie.path or "/")
                secure = bool(cookie.secure)
            except (AttributeError, TypeError, ValueError):
                domain, path, secure = _JAR_DOMAIN, "/", True
            per_attr[name] = {
                "domain": domain,
                "path": path,
                "secure": secure,
                "expires": str(expires) if expires else None,
            }
        expiries[psid] = per_exp
        attrs[psid] = per_attr
    return live, expiries, attrs


def pool_live_cookies(pool: AccountPool) -> list[dict[str, str]]:
    """Snapshot live cookie values from pool clients, one dict per account.

    Skips dead slots; keeps only jars carrying __Secure-1PSID identity.
    Best-effort: an unreadable jar is dropped, never raised.
    """
    live, _, _ = pool_live_state(pool)
    return live


def pool_live_expiries(pool: AccountPool) -> dict[str, dict[str, str]]:
    """Snapshot live expiry columns keyed by __Secure-1PSID then cookie name."""
    _, expiries, _ = pool_live_state(pool)
    return expiries


def _refresh_cookie_line(
    line: str,
    fresh: dict[str, str],
    expiries: dict[str, str],
) -> str:
    """Refresh matching jar lines to live value + live expiry when changed."""
    prefix = ""
    rest = line.strip()
    if rest.startswith("#HttpOnly"):
        prefix = "#HttpOnly "
        rest = rest[len("#HttpOnly") :].lstrip()
    parts = rest.split()
    if len(parts) < _JAR_COLUMNS:
        return line
    value = fresh.get(parts[5])
    if not value:
        return line
    if parts[6] == value:
        return line
    parts[6] = value
    if parts[5] in expiries:
        parts[4] = expiries[parts[5]]
    return prefix + "\t".join(parts)


def _refresh_expiry_line(line: str, expiries: dict[str, str]) -> str:
    """Refresh a stale expiry column when the live jar reports a newer one."""
    prefix = ""
    rest = line.strip()
    if rest.startswith("#HttpOnly"):
        prefix = "#HttpOnly "
        rest = rest[len("#HttpOnly") :].lstrip()
    parts = rest.split()
    if len(parts) < _JAR_COLUMNS:
        return line
    live_exp = expiries.get(parts[5])
    if live_exp is None or parts[4] == live_exp:
        return line
    try:
        if int(parts[4]) >= int(live_exp):
            return line
    except (TypeError, ValueError):
        return line
    parts[4] = live_exp
    return prefix + "\t".join(parts)


def _append_jar_line(name: str, value: str, attr: dict[str, object]) -> str:
    """Format one appended Netscape jar line from live jar attributes."""
    domain = str(attr.get("domain") or _JAR_DOMAIN)
    path = str(attr.get("path") or "/")
    secure = attr.get("secure")
    expires = attr.get("expires")
    flag = "FALSE" if secure is False else "TRUE"
    return f"\n{domain}\tTRUE\t{path}\t{flag}\t{expires or '0'}\t{name}\t{value}"


def sync_accounts_file(
    path: str | Path,
    live: list[dict[str, str]],
    expiries: dict[str, dict[str, str]] | None = None,
    attrs: dict[str, dict[str, dict[str, object]]] | None = None,
) -> int:
    """Persist live cookie jars back into accounts.txt jar blocks.

    Matches each ``` block to a live jar by __Secure-1PSID: refreshes lines
    whose value changed (expiry follows the live jar when known), refreshes
    stale expiry columns on unchanged values, and appends live names missing
    from the block with live domain/path/secure attributes. Unchanged lines
    pass through byte for byte. Atomic write; no-op when unchanged.
    Returns updated blocks. Blocks sharing one __Secure-1PSID refresh from
    the same live jar (duplicate jars stay identical, never cross-write).
    """
    file = Path(path)
    try:
        raw = file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    by_psid = {j["__Secure-1PSID"]: j for j in live if j.get("__Secure-1PSID")}
    if not by_psid:
        return 0
    exp_by_psid = expiries or {}
    attr_by_psid = attrs or {}
    updated = 0

    def _refresh_block(match: re.Match[str]) -> str:
        nonlocal updated
        block = match.group(1)
        psid = _parse_cookie_block(block).get("__Secure-1PSID", "")
        fresh = by_psid.get(psid)
        if not fresh:
            return match.group(0)
        have = _parse_cookie_block(block)
        exp = exp_by_psid.get(psid, {})
        attr = attr_by_psid.get(psid, {})
        lines = [_refresh_cookie_line(ln, fresh, exp) for ln in block.split("\n")]
        lines = [_refresh_expiry_line(ln, exp) for ln in lines]
        missing = sorted(n for n in fresh if n not in have)
        stripped = "\n".join(lines).rstrip("\n")
        extra = "".join(_append_jar_line(n, fresh[n], attr.get(n, {})) for n in missing)
        new_block = stripped + extra + ("\n" if block.endswith("\n") else "")
        if new_block == block:
            return match.group(0)
        updated += 1
        return "```\n" + new_block + "```"

    new_raw = _BLOCK_RE.sub(_refresh_block, raw)
    if not updated:
        return 0
    try:
        mode = file.stat().st_mode & 0o777
        tmp = file.with_name(file.name + ".tmp")
        tmp.write_text(new_raw, encoding="utf-8")
        tmp.chmod(mode)
        tmp.replace(file)
    except OSError:
        return 0
    return updated


@dataclass
class _Entry:
    client: GeminiClient
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    failures: int = 0
    live: bool = False


class AccountPool:
    """N load-balanced GeminiClients. One coroutine holds an account at a time."""

    def __init__(self, cookie_dicts: list[dict[str, str]]) -> None:
        """Build one Gemini client per cookie jar."""
        if not cookie_dicts:
            msg = "No Gemini accounts found (need __Secure-1PSID per block)"
            raise ValueError(msg)
        self._entries = [
            _Entry(
                GeminiClient(
                    secure_1psid=c.get("__Secure-1PSID"),
                    secure_1psidts=c.get("__Secure-1PSIDTS"),
                ),
            )
            for c in cookie_dicts
        ]
        self._rr = itertools.cycle(range(len(self._entries)))

    async def init_all(self) -> None:
        """Init every account; keep successes, mark failures cooled down.

        Raises only when zero accounts initialize (nothing to serve).
        """
        results = await asyncio.gather(
            *(e.client.init(timeout=450) for e in self._entries),
            return_exceptions=True,
        )
        ok = 0
        for entry, result in zip(self._entries, results, strict=True):
            if isinstance(result, BaseException):
                entry.failures = _COOLDOWN_STRIKES
            else:
                entry.live = True
                entry.failures = 0
                ok += 1
        if not ok:
            msg = "No Gemini accounts initialized (all cookie jars failed)"
            raise RuntimeError(msg)

    async def build_replacement(
        self,
        index: int,
        cookies: dict[str, str],
    ) -> GeminiClient | None:
        """Init a fresh client off to the side; None when unusable.

        Network init runs WITHOUT the per-account lock so pinned turns keep
        flowing on the old client. The returned client is unpublished: pass
        it to swap_client under lock_for(index). Failed builds are closed
        before returning, so no background tasks leak.
        """
        _ = self._entries[index]
        fresh = GeminiClient(
            secure_1psid=cookies.get("__Secure-1PSID"),
            secure_1psidts=cookies.get("__Secure-1PSIDTS"),
        )
        try:
            await fresh.init(timeout=450)
        except BaseException:  # noqa: BLE001 - CancelledError must also close fresh
            with suppress(Exception):
                await fresh.close()
            return None
        if fresh.account_status != AccountStatus.AVAILABLE:
            with suppress(Exception):
                await fresh.close()
            return None
        return fresh

    def swap_client(self, index: int, fresh: GeminiClient) -> None:
        """Publish an initialized replacement; close the old client.

        Callers MUST hold lock_for(index). Turns that grabbed the lock first
        already hold the old client object and finish on it; later turns see
        the fresh client. The old session closes in the background so its
        teardown never blocks the lock. Re-check AVAILABLE under the lock
        before calling; a concurrently recovered account must skip (and the
        caller must close the unused fresh client itself).
        """
        entry = self._entries[index]
        old = entry.client
        entry.client = fresh
        entry.live = True
        entry.failures = 0
        asyncio.get_running_loop().create_task(self._close_replaced(old))

    @staticmethod
    async def _close_replaced(old: GeminiClient) -> None:
        """Close a swapped-out client; never raises into the task owner."""
        with suppress(Exception):
            await old.close()

    async def close_all(self) -> None:
        """Close every initialized account client."""
        await asyncio.gather(
            *(e.client.close() for e in self._entries if e.live),
            return_exceptions=True,
        )

    def client_count(self) -> int:
        """Return the number of configured accounts."""
        return len(self._entries)

    def has_slot(self, index: int | None) -> bool:
        """Check whether index names a live pool slot (sticky pin target)."""
        return isinstance(index, int) and 0 <= index < len(self._entries)

    def client_at(self, index: int) -> GeminiClient:
        """Return the Gemini client at a pool index."""
        return self._entries[index].client

    def pick(self) -> tuple[int, GeminiClient]:
        """Round-robin pick, skipping cooled-down accounts; all cooled -> reset."""
        for _ in range(len(self._entries)):
            i = next(self._rr)
            if self._entries[i].failures < _COOLDOWN_STRIKES:
                return i, self._entries[i].client
        # Every account cooled down: forgive next in rotation so recovery works.
        i = next(self._rr)
        self._entries[i].failures = 0
        return i, self._entries[i].client

    def pick_available(self) -> tuple[int, GeminiClient] | None:
        """Round-robin pick restricted to AVAILABLE, cooldown-aware slots.

        Returns None when no authenticated, non-cooled-down slot exists, so
        callers can distinguish "no auth anywhere" (retry text path) from a
        routable account. Cooldown strikes still steer fresh conversations
        away from failing accounts; the shared round-robin cursor keeps load
        balanced. Does NOT reset an all-cooled-down pool: failure recovery
        stays with pick().
        """
        for _ in range(len(self._entries)):
            i = next(self._rr)
            entry = self._entries[i]
            if entry.failures >= _COOLDOWN_STRIKES:
                continue
            if entry.client.account_status != AccountStatus.AVAILABLE:
                continue
            return i, entry.client
        return None

    def lock_for(self, index: int) -> asyncio.Lock:
        """Return the per-account lock serializing one account's turns."""
        return self._entries[index].lock

    def report(self, index: int, *, ok: bool) -> None:
        """Record success (reset strikes) or failure (add one strike)."""
        e = self._entries[index]
        e.failures = 0 if ok else e.failures + 1

    def models(self) -> list[dict[str, object]]:
        """List deduplicated registry models across accounts."""
        seen: dict[str, dict[str, object]] = {}
        for e in self._entries:
            for m in e.client.list_models() or []:
                seen.setdefault(
                    m.model_name,
                    {
                        "id": m.model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "gemini",
                    },
                )
        return [seen[k] for k in sorted(seen)]

    def resolve(self, name: str | None) -> str | None:
        """Resolve an OpenAI-style model name to a Gemini registry model."""
        if not name:
            return None
        try:
            return self._entries[0].client.resolve_model(name).model_name
        except ValueError:
            return None

    def available_model(self, name: str | None) -> str | None:
        """Resolve to a session-selectable registry name, else None for default."""
        if not name:
            return None
        for e in self._entries:
            try:
                direct = e.client.resolve_model(name)
            except ValueError:
                continue
            if direct.is_available:
                return direct.model_name
        for e in self._entries:
            for m in e.client.list_models() or []:
                if m.is_available:
                    return m.model_name
        return None

    def auth_state(self) -> str:
        """Report "ok" when any account is authenticated, else "degraded"."""
        for e in self._entries:
            if e.client.account_status == AccountStatus.AVAILABLE:
                return "ok"
        return "degraded"

    def display_slugs(self) -> list[str]:
        """UI-derived ids (e.g. gemini-3.8-flash) for the /v1/models listing."""
        slugs: list[str] = []
        for e in self._entries:
            for m in e.client.list_models() or []:
                slug = "gemini-" + re.sub(r"\s+", "-", m.display_name.strip().lower())
                if slug and slug not in slugs:
                    slugs.append(slug)
        return slugs
