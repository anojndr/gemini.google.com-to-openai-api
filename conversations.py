"""Session store: conversation keys/aliases -> account-bound Gemini continuation.

Continuation is stored as plain Gemini metadata ([cid, rid, rcid]), so any
request can resume native server-side history with a fresh ChatSession on the
same account. Keys come from Response conversation ids, previous_response_id
chains, or content fingerprints of Chat Completions prefixes.
"""

from __future__ import annotations

import asyncio


class SessionState:
    __slots__ = ("account", "lock", "metadata", "norm")

    def __init__(self) -> None:
        self.account: int | None = None
        self.metadata: list = []
        self.norm: list[str] = []
        self.lock = asyncio.Lock()


class SessionStore:
    def __init__(self, cap: int = 2000) -> None:
        self._states: dict[str, SessionState] = {}
        self._aliases: dict[str, str] = {}
        self._responses: dict[str, dict] = {}
        self._guard = asyncio.Lock()
        self._cap = cap

    def _resolve(self, key: str) -> str:
        for _ in range(5):
            if key not in self._aliases:
                break
            key = self._aliases[key]
        return key

    async def get(self, key: str) -> SessionState | None:
        async with self._guard:
            return self._states.get(self._resolve(key))

    async def get_or_new(self, key: str) -> tuple[SessionState, bool]:
        async with self._guard:
            key = self._resolve(key)
            state = self._states.get(key)
            if state is None:
                if len(self._states) >= self._cap:
                    self._states.pop(next(iter(self._states)))
                state = SessionState()
                self._states[key] = state
                return state, True
            return state, False

    async def lock_for(self, key: str) -> asyncio.Lock:
        """Per-conversation lock serializing read-modify-write of one session."""
        state, _ = await self.get_or_new(key)
        return state.lock

    async def link(self, alias: str, key: str) -> None:
        async with self._guard:
            self._aliases[alias] = self._resolve(key)
            if len(self._aliases) > self._cap * 2:
                self._aliases.pop(next(iter(self._aliases)))

    async def set_norm(self, key: str, norm: list[str]) -> None:
        async with self._guard:
            state = self._states.get(self._resolve(key))
            if state is not None:
                state.norm = list(norm)
    async def get_norm(self, key: str) -> list[str] | None:
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
            return state

    async def save_response(self, rid: str, obj: dict) -> None:
        async with self._guard:
            if len(self._responses) >= self._cap:
                self._responses.pop(next(iter(self._responses)))
            self._responses[rid] = obj

    async def get_response(self, rid: str) -> dict | None:
        async with self._guard:
            return self._responses.get(rid)
