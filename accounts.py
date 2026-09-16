# Copyright 2026 gem2oai contributors.
"""Account pool: parse Netscape cookie blocks, round-robin GeminiClients.

accounts.txt layout: free text with one Netscape cookie jar per ``` block.
Any number of blocks supported; each block needs __Secure-1PSID.
"""

from __future__ import annotations

import asyncio
import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

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
                ok += 1
        if not ok:
            msg = "No Gemini accounts initialized (all cookie jars failed)"
            raise RuntimeError(msg)

    async def close_all(self) -> None:
        """Close every initialized account client."""
        await asyncio.gather(
            *(e.client.close() for e in self._entries if e.live),
            return_exceptions=True,
        )

    def client_count(self) -> int:
        """Return the number of configured accounts."""
        return len(self._entries)

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
