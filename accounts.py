"""Account pool: parse Netscape cookie blocks, round-robin GeminiClients.

accounts.txt layout: free text with one Netscape cookie jar per ``` block.
Any number of blocks supported; each block needs __Secure-1PSID.
"""

from __future__ import annotations

import asyncio
import itertools
import re
from dataclasses import dataclass, field

from gemini_webapi import GeminiClient

_BLOCK_RE = re.compile(r"```\n(.*?)```", re.S)


def _parse_cookie_block(block: str) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for line in block.splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0].startswith("."):
            cookies[parts[5]] = parts[6]
    return cookies


def load_account_cookies(path: str) -> list[dict[str, str]]:
    """Extract cookie dicts from every ``` jar block; skip blocks without auth."""
    raw = open(path, encoding="utf-8", errors="replace").read()
    blocks = _BLOCK_RE.findall(raw)
    accounts = [_parse_cookie_block(b) for b in blocks]
    return [c for c in accounts if c.get("__Secure-1PSID")]


@dataclass
class _Entry:
    client: GeminiClient
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    failures: int = 0


class AccountPool:
    """N load-balanced GeminiClients. One coroutine holds an account at a time."""

    def __init__(self, cookie_dicts: list[dict[str, str]]):
        if not cookie_dicts:
            raise ValueError("No Gemini accounts found (need __Secure-1PSID per block)")
        self._entries = [
            _Entry(
                GeminiClient(
                    secure_1psid=c.get("__Secure-1PSID"),
                    secure_1psidts=c.get("__Secure-1PSIDTS"),
                )
            )
            for c in cookie_dicts
        ]
        self._rr = itertools.cycle(range(len(self._entries)))

    async def init_all(self) -> None:
        await asyncio.gather(*(e.client.init(timeout=450) for e in self._entries))

    async def close_all(self) -> None:
        await asyncio.gather(*(e.client.close() for e in self._entries))

    def pick(self) -> tuple[int, GeminiClient]:
        """Round-robin pick, skipping accounts in cooldown (3 consecutive failures)."""
        for _ in range(len(self._entries)):
            i = next(self._rr)
            if self._entries[i].failures < 3:
                return i, self._entries[i].client
        i = next(self._rr)
        return i, self._entries[i].client

    def lock_for(self, index: int) -> asyncio.Lock:
        return self._entries[index].lock

    def report(self, index: int, ok: bool) -> None:
        e = self._entries[index]
        e.failures = 0 if ok else e.failures + 1

    def models(self) -> list[dict]:
        seen: dict[str, dict] = {}
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
        return sorted(seen.values(), key=lambda m: m["id"])

    def resolve(self, name: str | None) -> str | None:
        """Resolve an OpenAI-style model name to a Gemini model on account 0's registry."""
        if not name:
            return None
        try:
            return self._entries[0].client.resolve_model(name).model_name
        except ValueError:
            return None

    def display_slugs(self) -> list[str]:
        """UI-derived ids (e.g. gemini-3.8-flash) for the /v1/models listing."""
        slugs: list[str] = []
        for e in self._entries:
            for m in e.client.list_models() or []:
                slug = "gemini-" + re.sub(r"\s+", "-", m.display_name.strip().lower())
                if slug and slug not in slugs:
                    slugs.append(slug)
        return slugs
