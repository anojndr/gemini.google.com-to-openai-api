# Copyright 2026 gem2oai contributors.
"""gemini.google.com -> OpenAI-compatible API (Chat Completions + Responses).

Backend: gemini-webapi over direct HTTPS (curl_cffi, no browser).
Multi-turn: native Gemini ChatSession per conversation; only new turns are sent.
Accounts: N cookie jars in accounts.txt, round-robin + failover.
Image output: Gemini GeneratedImage -> freeimage.host URL -> markdown.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from gemini_webapi.constants import AccountStatus
from gemini_webapi.exceptions import (
    APIError,
    AuthError,
    GeminiError,
)
from gemini_webapi.types import GeneratedImage, Image, ModelOutput
from starlette.datastructures import UploadFile as StarletteUploadFile

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from gemini_webapi import ChatSession

import chrome_cookies
import freeimage
from accounts import (
    AccountPool,
    GeminiClient,
    load_account_cookies,
    pool_live_state,
    sync_accounts_file,
)
from config import ACCOUNTS_FILE, BASE_DIR, DB_PATH, PORT, REDIS_URL, freeimage_api_key
from content import UploadFile, UploadStore, message_to_turn, part_identity
from conversations import SessionStore
from redis_store import RedisState

_log = logging.getLogger("gem2oai")
logging.basicConfig(level=logging.INFO, format="%(message)s")

Json = dict[str, Any]
JsonList = list[dict[str, Any]]
HandlerResult = Json | JSONResponse | StreamingResponse

THINKING_ALIASES = {
    "gemini-3.8-flash-thinking",
    "gemini-3.8-flash-extended-thinking",
    "gemini-3.8-flash-et",
}

IMG_TMP = Path(tempfile.gettempdir()) / "gem2oai-imgs"
IMG_TMP.mkdir(parents=True, exist_ok=True)


def _now() -> int:
    return int(time.time())


def _new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------- models ---


def _resolve_model(pool: AccountPool, name: str | None) -> tuple[str | None, bool]:
    """OpenAI model name -> (gemini registry name, extended_thinking).

    Names resolve through session availability: an expired/unauthenticated
    session only offers its guest default, so a request for an unavailable
    model falls back to a selectable one (or None for Google's default)
    instead of failing the turn.
    """
    n = (name or "").strip().lower()
    thinking = "think" in n or n.endswith("-et")
    if not n or n in ("default", "auto"):
        return None, thinking
    if n in THINKING_ALIASES:
        return pool.available_model("gemini-flash-lite"), True
    core = n.replace("extended", "").replace("thinking", "").replace("-et", "")
    if "lite" in core or "3.5" in core:
        return pool.available_model("gemini-flash-lite"), thinking
    if "pro" in core or "3.1" in core:
        return pool.available_model("gemini-pro"), thinking
    if "flash" in core or "3.8" in core or "3.6" in core:
        return pool.available_model("gemini-flash"), thinking
    return pool.available_model(name), thinking


def _model_list(pool: AccountPool) -> list[Json]:
    """List registry models plus display-slug and flash aliases, sorted by id."""
    seen: dict[str, Json] = {str(m["id"]): m for m in pool.models()}
    for slug in pool.display_slugs():
        seen.setdefault(
            slug,
            {"id": slug, "object": "model", "created": 0, "owned_by": "gemini"},
        )
    for alias in ["gemini-3.8-flash", *sorted(THINKING_ALIASES)]:
        seen.setdefault(
            alias,
            {"id": alias, "object": "model", "created": 0, "owned_by": "gemini"},
        )
    return [seen[k] for k in sorted(seen)]


def _chat_completion(
    model_name: str | None,
    key: str,
    text: str,
    thoughts: str,
    prompt_for_usage: str,
) -> dict[str, Any]:
    """Build a buffered chat-completion response object."""
    msg: Json = {"role": "assistant", "content": text}
    if thoughts:
        msg["reasoning_content"] = thoughts
    return {
        "id": _new_id("chatcmpl-"),
        "object": "chat.completion",
        "created": _now(),
        "model": model_name or "gemini-flash",
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": "stop",
            },
        ],
        "usage": _usage(prompt_for_usage, text),
        "conversation_id": key.removeprefix("conv:")
        if key.startswith("conv:")
        else None,
    }


def _chat_chunk(
    cid: str,
    model_name: str | None,
    delta: Json,
    usage: Json | None = None,
) -> str:
    """Encode one chat-completion chunk as an SSE frame."""
    choice: Json = {"index": 0, "delta": delta, "finish_reason": None}
    if usage is not None:
        choice["finish_reason"] = "stop"
    body: Json = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model_name or "gemini-flash",
        "choices": [choice],
    }
    if usage is not None:
        body["usage"] = usage
    return _sse(body)


async def _chat_delta_frames(
    cid: str,
    model_name: str | None,
    payload: str,
    state: Json,
) -> AsyncIterator[str]:
    """Yield role (once) plus content frames for one text delta."""
    if not state["sent_role"]:
        state["sent_role"] = True
        yield _chat_chunk(cid, model_name, {"role": "assistant"})
    state["full"] += payload
    yield _chat_chunk(cid, model_name, {"content": payload})


class ChatDoneContext(NamedTuple):
    """Context for persisting chat state and yielding done frames."""

    cid: str
    key: str
    account: int
    new_meta: list[str | None]
    model_name: str | None
    prompt_for_usage: str
    out: ModelOutput


async def _chat_done_frames(
    http: httpx.AsyncClient,
    sessions: SessionStore,
    ctx: ChatDoneContext,
    state: Json,
) -> AsyncIterator[str]:
    """Persist completion state and yield image/stop frames for one done event."""
    md = await _gemini_images_to_markdown(http, ctx.out)
    if md:
        state["full"] += md
        yield _chat_chunk(
            cid=ctx.cid,
            model_name=ctx.model_name,
            delta={"content": md},
        )
    await _save_state(
        sessions,
        ctx.key,
        ctx.account,
        ctx.new_meta,
    )
    yield _chat_chunk(
        cid=ctx.cid,
        model_name=ctx.model_name,
        delta={},
        usage=_usage(ctx.prompt_for_usage, state["full"]),
    )


async def _boot_state(app: FastAPI) -> tuple[AccountPool, int, int]:
    """Boot accounts plus SQLite/Redis stores; return (pool, count, purged)."""
    cookies = load_account_cookies(str(ACCOUNTS_FILE))
    count = len(cookies)
    pool = AccountPool(cookies)
    try:
        await pool.init_all()
    except RuntimeError as exc:
        _log.exception("gem2oai: FATAL: %s", exc)
        msg = str(exc)
        raise RuntimeError(msg) from exc
    sessions = SessionStore(path=DB_PATH)
    try:
        UploadStore.bind(DB_PATH)
    except OSError:
        sessions.close()
        await pool.close_all()
        raise
    redis_state = await RedisState.create(REDIS_URL)
    sessions.attach_redis(redis_state)
    UploadStore.attach_redis(redis_state)
    app.state.pool = pool
    app.state.sessions = sessions
    app.state.redis = redis_state
    app.state.http = httpx.AsyncClient(timeout=120, follow_redirects=True)
    dropped = await _purge_dead_continuations(pool, sessions)
    return pool, count, dropped


async def _shutdown_state(app: FastAPI, pool: AccountPool) -> None:
    """Stop cookie sync, persist cookies, and close pool/stores/Redis."""
    await app.state.http.aclose()
    try:
        live, expiries, attrs = pool_live_state(pool)
        sync_accounts_file(ACCOUNTS_FILE, live, expiries, attrs)
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        _log.warning("gem2oai: shutdown cookie sync skipped: %s", exc)
    await pool.close_all()
    app.state.sessions.close()
    UploadStore.close()
    UploadStore.attach_redis(None)
    redis_state = getattr(app.state, "redis", None)
    if redis_state is not None:
        await redis_state.aclose()
        app.state.redis = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Boot accounts, SQLite stores, and shared HTTP client; close on shutdown."""
    os.environ.setdefault("GEMINI_COOKIE_PATH", str(BASE_DIR / ".gemini-cookie-cache"))
    Path(os.environ["GEMINI_COOKIE_PATH"]).mkdir(parents=True, exist_ok=True)
    for stale in Path(os.environ["GEMINI_COOKIE_PATH"]).glob(".cached_cookies_*.json"):
        with suppress(OSError):
            stale.unlink()
    pool, count, dropped = await _boot_state(app)
    # Chrome is the source of truth, but only for values the live session
    # has not rotated past: the clients rotate 1PSIDTS server-side every
    # ~10 min, ahead of what the browser holds. Overwriting the file with
    # older Chrome values would downgrade working credentials, so refresh
    # from Chrome first, then let live values win on every conflict.
    try:
        chrome_synced = chrome_cookies.refresh_accounts_from_chrome(ACCOUNTS_FILE)
    except Exception as exc:  # noqa: BLE001 - Chrome read must never block boot
        _log.warning("gem2oai: Chrome cookie refresh skipped: %s", exc)
        chrome_synced = 0
    try:
        live, expiries, attrs = pool_live_state(pool)
        synced = sync_accounts_file(ACCOUNTS_FILE, live, expiries, attrs)
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        _log.warning("gem2oai: startup cookie sync skipped: %s", exc)
        synced = 0
    cookie_task = asyncio.create_task(_cookie_sync_loop(pool))
    _log.info(
        "gem2oai: %d account(s) ready, freeimage=%s, purged=%d, "
        "cookies_synced=%d, chrome_synced=%d",
        count,
        "yes" if freeimage_api_key() else "NO KEY",
        dropped,
        synced,
        chrome_synced,
    )
    yield
    cookie_task.cancel()
    with suppress(asyncio.CancelledError):
        await cookie_task
    await _shutdown_state(app, pool)


app = FastAPI(title="gemini.google.com-to-openai-api", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pool(req: Request) -> AccountPool:
    return req.app.state.pool


def _sessions(req: Request) -> SessionStore:
    return req.app.state.sessions


def _http(req: Request) -> httpx.AsyncClient:
    return req.app.state.http


def _err(status: int, message: str, code: str = "server_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": code, "code": code}},
    )


def _is_upstream_error(exc: BaseException) -> bool:
    """Check whether the failure implicates the Gemini account or network."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if any(
        k in name
        for k in (
            "Timeout",
            "Auth",
            "UsageLimit",
            "Blocked",
            "APIError",
            "GeminiError",
        )
    ):
        return True
    return any(
        k in msg
        for k in (
            "timed out",
            "timeout",
            "429",
            "503",
            "500",
            "unauth",
            "auth",
            "quota",
            "blocked",
            "network",
            "connection",
            "reset",
        )
    )


def _map_exception(exc: BaseException) -> tuple[int, str]:
    """Map an upstream failure to an (HTTP status, message) pair."""
    name = type(exc).__name__
    msg = str(exc) or name
    if "UsageLimitExceeded" in name or "429" in msg:
        return 429, msg
    if "Timeout" in name or "timed out" in msg.lower():
        return 504, msg
    if "Auth" in name or "unauth" in msg.lower():
        return 502, msg
    return 500, msg


async def _purge_dead_continuations(pool: AccountPool, sessions: SessionStore) -> int:
    """Drop stored continuations whose account cannot offer them.

    Guest-era cids were minted by sessions that could not read history;
    resuming them against the new authenticated sessions stalls until the
    watchdog fires. Any stored row bound to a non-AVAILABLE account is dead
    weight: drop it so the next request starts fresh instead of hanging.
    Rows pinned to a pool slot that no longer exists (accounts removed)
    would resume a foreign cid on the wrong Google session after re-pin,
    so they are dropped too. Returns the number of rows dropped.
    """
    try:
        rows = await sessions.snapshot_all()
    except (AttributeError, TypeError, ValueError):
        return 0
    dropped = 0
    for key, account, metadata in rows:
        if not metadata:
            continue
        if account is None or not pool.has_slot(account):
            try:
                if await sessions.drop_if(key, account, metadata):
                    dropped += 1
            except (AttributeError, TypeError, ValueError, OSError):
                continue
            continue
        if pool.client_at(account).account_status == AccountStatus.AVAILABLE:
            continue
        try:
            if await sessions.drop_if(key, account, metadata):
                dropped += 1
        except (AttributeError, TypeError, ValueError, OSError):
            continue
    return dropped


def _is_resume_error(exc: BaseException) -> bool:
    """Check whether a failure could implicate a dead continuation id.

    Only resumed turns reach the library's history-recovery path: a fresh
    cid streams or fails fast, while a dead guest-era cid stalls until the
    watchdog fires and recovery polling times out. Callers gate on
    request.metadata, so any error here is retryable fresh.
    """
    msg = str(exc).lower()
    return any(
        k in msg
        for k in (
            "recovery timed out",
            "turn timed out",
            "stream stalled",
            "silently aborted",
            "cannot read history",
            "polling for c_",
            "no cid found",
            "stream suspended",
        )
    )


# ------------------------------------------------------------ gemini run ---

TurnUpload = tuple[UploadFile, str]
Turn = tuple[str, list[TurnUpload]]
StreamDelta = tuple[str, str]
StreamDone = tuple[str, tuple[ModelOutput | None, list[str | None]]]


def _uploads_need_auth(turns: list[Turn]) -> bool:
    """Check whether any turn stages file uploads for Gemini."""
    return any(uploads for _, uploads in turns)


def _account_is_upload_capable(pool: AccountPool, account: int) -> bool:
    """Check whether an account session can take file-attached turns."""
    try:
        client = pool.client_at(account)
    except (IndexError, TypeError, ValueError):
        return False
    return client.account_status == AccountStatus.AVAILABLE


def _ensure_upload_capable(
    client: GeminiClient,
    account: int,
    turns: list[Turn],
) -> None:
    """Fail fast when uploads need an authenticated session.

    gemini-webapi gates every file-attached turn on AVAILABLE, and a guest
    session never completes one (the raw upload succeeds but the reply
    stalls). Raise an actionable AuthError here and clean staged files
    instead of surfacing the library's generic "Permission denied".
    """
    if not _uploads_need_auth(turns):
        return
    if client.account_status == AccountStatus.AVAILABLE:
        return
    for _, uploads in turns:
        for up, _ in uploads:
            up.cleanup()
    msg = (
        "image/file attachments need an authenticated Gemini session "
        f"(account {account} is {client.account_status.name}); "
        "text-only prompts still work. Re-export fresh cookie jars into "
        "accounts.txt and run ./restart.sh, then retry."
    )
    raise AuthError(msg)


async def _send_turn(
    chat: ChatSession,
    text: str,
    uploads: list[TurnUpload],
    *,
    thinking: bool,
) -> ModelOutput:
    """Send one text turn plus staged file paths; always clean up staging."""
    files: list[str] | None = [str(up.path) for up, _ in uploads] or None
    try:
        return await chat.send_message(
            text.strip() or " ",
            files=files,  # ty: ignore[invalid-argument-type]
            extended_thinking=thinking,
        )
    finally:
        for up, _ in uploads:
            up.cleanup()


class RunRequest(NamedTuple):
    """Everything needed to replay turns through one Gemini ChatSession."""

    account: int
    model: str | None
    thinking: bool
    turns: list[Turn]
    metadata: list[str | None]


async def _send_turn_bounded(
    chat: ChatSession,
    text: str,
    uploads: list[TurnUpload],
    *,
    thinking: bool,
    label: str,
) -> ModelOutput:
    """Send one turn, failing with TimeoutError after 100s without a reply."""
    try:
        return await asyncio.wait_for(
            _send_turn(chat, text, uploads, thinking=thinking),
            timeout=100,
        )
    except asyncio.TimeoutError as exc:
        msg = f"{label} timed out after 100s: {exc}"
        raise TimeoutError(msg) from exc


async def _run_turns(
    pool: AccountPool,
    request: RunRequest,
) -> tuple[ModelOutput, list[str | None]]:
    """Replay turns through one native ChatSession. Returns (final output, metadata)."""
    client = (
        _client_at(pool, request.account) if request.account >= 0 else pool.pick()[1]
    )
    chat = client.start_chat(
        model=request.model,
        **({"metadata": request.metadata} if request.metadata else {}),
    )
    out: ModelOutput | None = None
    try:
        _ensure_upload_capable(client, request.account, request.turns)
        async with pool.lock_for(request.account):
            for text, uploads in request.turns:
                out = await _send_turn_bounded(
                    chat,
                    text,
                    uploads,
                    thinking=request.thinking,
                    label="turn",
                )
        pool.report(request.account, ok=True)
        return _require_output(out, chat)
    except ValueError:
        pool.report(request.account, ok=True)
        raise
    except (
        GeminiError,
        AuthError,
        APIError,
        httpx.HTTPError,
        OSError,
        TimeoutError,
    ) as exc:
        pool.report(request.account, ok=not _is_upstream_error(exc))
        if request.metadata and _is_resume_error(exc):
            return await _run_turns(pool, request._replace(metadata=[]))
        raise


def _require_output(
    out: ModelOutput | None,
    chat: ChatSession,
) -> tuple[ModelOutput, list[str | None]]:
    """Return the turn output and metadata, rejecting empty turn lists."""
    if out is None:
        msg = "no turns to send"
        raise ValueError(msg)
    return out, list(chat.metadata)


def _client_at(pool: AccountPool, account: int) -> GeminiClient:
    """Return the Gemini client at a pool index."""
    return pool.client_at(account)


async def _stream_attempt(
    pool: AccountPool,
    chat: ChatSession,
    lock: asyncio.Lock,
    request: RunRequest,
) -> AsyncIterator[StreamDelta | StreamDone]:
    """Stream one attempt's turns; yields deltas then done with metadata."""
    await lock.acquire()
    try:
        out: ModelOutput | None = None
        for text, uploads in request.turns[:-1]:
            out = await _send_turn_bounded(
                chat,
                text,
                uploads,
                thinking=request.thinking,
                label="replay turn",
            )
        text, uploads = request.turns[-1]
        files: list[str] | None = [str(up.path) for up, _ in uploads] or None
        try:
            stream = chat.send_message_stream(
                text.strip() or " ",
                files=files,  # ty: ignore[invalid-argument-type]
                extended_thinking=request.thinking,
            )
            first: ModelOutput | None = await asyncio.wait_for(
                stream.__anext__(),
                timeout=100,
            )
            out = first
            if first.text_delta:
                yield ("delta", first.text_delta)
            while True:
                try:
                    chunk = await asyncio.wait_for(stream.__anext__(), timeout=100)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    msg = f"stream stalled after last delta: {exc}"
                    raise TimeoutError(msg) from exc
                out = chunk
                if chunk.text_delta:
                    yield ("delta", chunk.text_delta)
        except asyncio.TimeoutError as exc:
            msg = f"stream stalled: {exc}"
            raise TimeoutError(msg) from exc
        finally:
            for up, _ in uploads:
                up.cleanup()
        pool.report(request.account, ok=True)
        yield ("done", (out, list(chat.metadata)))
    finally:
        lock.release()


async def _run_turns_stream(
    pool: AccountPool,
    request: RunRequest,
) -> AsyncIterator[StreamDelta | StreamDone]:
    """Replay prior turns, then stream the final turn's deltas.

    Yields ("delta", text) chunks, then ("done", (output, metadata)).
    Holds the account lock for the whole conversation update.
    Raises ValueError when turns is empty (caller must 400, not stream).
    """
    if not request.turns:
        msg = "no turns to send"
        raise ValueError(msg)
    client = _client_at(pool, request.account)
    lock = pool.lock_for(request.account)
    try:
        _ensure_upload_capable(client, request.account, request.turns)
        chat = client.start_chat(
            model=request.model,
            **({"metadata": request.metadata} if request.metadata else {}),
        )
        async for item in _stream_attempt(pool, chat, lock, request):
            yield item
    except ValueError:
        pool.report(request.account, ok=True)
        raise
    except (
        GeminiError,
        AuthError,
        APIError,
        httpx.HTTPError,
        OSError,
        TimeoutError,
    ) as exc:
        if not request.metadata or not _is_resume_error(exc):
            pool.report(request.account, ok=not _is_upstream_error(exc))
            raise
        pool.report(request.account, ok=True)
        chat = client.start_chat(model=request.model)
        async for item in _stream_attempt(pool, chat, lock, request):
            yield item


# ---------------------------------------------------------------- images ---


async def _rehost_image(
    http: httpx.AsyncClient,
    img: Image,
    idx: int,
) -> str | None:
    """Save one Gemini image and re-host it; fall back to the Gemini URL."""
    url = img.url or ""
    try:
        path = await img.save(
            path=str(IMG_TMP),
            filename=f"gen_{uuid.uuid4().hex[:8]}",
        )
    except (OSError, ValueError, RuntimeError, httpx.HTTPError):
        return f"![image]({url})" if url else None
    try:
        data = await asyncio.to_thread(Path(path).read_bytes)
    except (OSError, ValueError, RuntimeError):
        return f"![image]({url})" if url else None
    finally:
        await asyncio.to_thread(Path(path).unlink, missing_ok=True)
    public = await freeimage.upload_png(http, data, f"gemini_{idx}.png")
    return f"![Generated Image {idx}]({public or url})"


async def _gemini_images_to_markdown(
    http: httpx.AsyncClient,
    out: ModelOutput,
) -> str:
    """Download generated/web images, re-host on freeimage, return markdown block."""
    imgs = list(out.images or [])
    if not imgs:
        return ""
    md = [
        line
        for line in await asyncio.gather(
            *(_rehost_image(http, img, i) for i, img in enumerate(imgs)),
        )
        if line
    ]
    return ("\n\n" + "\n\n".join(md)) if md else ""


def _full_text(out: ModelOutput) -> str:
    """Return the completed text of a Gemini output."""
    return out.text or ""


def _thoughts(out: ModelOutput) -> str:
    """Return the extended-thinking trace of a Gemini output, if any."""
    return out.thoughts or ""


def _usage(prompt: str, completion: str) -> dict[str, int]:
    """Estimate OpenAI-style token usage from character counts."""
    pt, ct = max(1, len(prompt) // 4), max(1, len(completion) // 4)
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


# ------------------------------------------------- conversation chaining ---


def _fingerprint(messages: JsonList) -> str:
    """Hash chat messages (text inline, files by identity) for replay dedup."""
    parts: list[str] = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            parts.append(f"{m.get('role', '')}:{c}")
        else:
            inner = "|".join(
                (
                    p.get("text", "")
                    if isinstance(p, dict)
                    and p.get("type") in ("text", "input_text", "output_text")
                    else part_identity(p)
                )
                for p in (c or [])
            )
            parts.append(f"{m.get('role', '')}:{inner}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:24]


def _prefix_turn(role: str, text: str, uploads: list[TurnUpload]) -> Turn:
    """Tag a system turn; other roles pass through unchanged."""
    prefix = "[System] " if role == "system" else ""
    return (f"{prefix}{text}", uploads)


async def _fresh_turns(
    http: httpx.AsyncClient,
    messages: JsonList,
) -> list[Turn]:
    """Convert unseen-history messages to user/system turns for a new session."""
    turns: list[Turn] = []
    for m in messages:
        role, text, uploads = await message_to_turn(http, m)
        if role in ("assistant", "tool"):
            continue
        turns.append(_prefix_turn(role, text, uploads))
    return turns or [(" ", [])]


async def _chat_turns(
    http: httpx.AsyncClient,
    sessions: SessionStore,
    pool: AccountPool,
    messages: JsonList,
    conversation_id: str | None,
) -> tuple[str, list[Turn], list[str | None]]:
    """Resolve (state_key, new turns, resume metadata) for a Chat request.

    Chained requests send ONLY the new trailing message(s); unseen histories
    replay user/system turns in order to build native Gemini history.
    """
    if conversation_id:
        key = f"conv:{conversation_id}"
        state, created = await sessions.get_or_new(key)
        if not created and state.metadata:
            last = messages[-1:]
            turns = []
            for m in last:
                role, text, uploads = await message_to_turn(http, m)
                if role in ("assistant", "tool"):
                    for up, _ in uploads:
                        up.cleanup()
                    msg = "last message has role assistant/tool; nothing new to send"
                    raise ValueError(msg)
                turns.append(_prefix_turn(role, text, uploads))
            return key, turns, state.metadata
        return key, await _fresh_turns(http, messages), []

    if len(messages) > 1:
        prev_fp = _fingerprint(messages[:-1])
        prev = await sessions.get(f"fp:{prev_fp}")
        if prev is not None and prev.metadata and pool.has_slot(prev.account):
            key = f"fp:{_fingerprint(messages)}"
            state, _ = await sessions.get_or_new(key)
            if not state.metadata:
                state.account = prev.account
                state.metadata = prev.metadata
                await sessions.persist(key)
                resume_meta = list(prev.metadata)
            else:
                resume_meta = list(state.metadata)
            role, text, uploads = await message_to_turn(http, messages[-1])
            return key, [_prefix_turn(role, text, uploads)], resume_meta

    key = f"fp:{_fingerprint(messages)}"
    await sessions.get_or_new(key)
    return key, await _fresh_turns(http, messages), []


async def _pick_account(
    pool: AccountPool,
    sessions: SessionStore,
    key: str,
) -> tuple[int, list[str | None]]:
    """Sticky account per conversation state; fresh round-robin otherwise.

    A pin survives only while its pool slot exists; an account index from a
    resized pool (or a row written before an account was added) re-pins
    through round-robin so the resumed Gemini cid is never sent to the
    wrong Google session.
    """
    state = await sessions.get(key)
    if state is not None and state.account is not None and pool.has_slot(state.account):
        account = state.account
        return account, state.metadata
    idx, _ = pool.pick()
    st, _ = await sessions.get_or_new(key)
    if st.account is None or not pool.has_slot(st.account):
        # Fresh key or stale pin (pool resized): pin without touching any
        # live continuation metadata the convo lock owner may hold.
        st.account = idx
        await sessions.persist(key)
    return idx, st.metadata


async def _pick_account_for_turns(
    pool: AccountPool,
    sessions: SessionStore,
    key: str,
    turns: list[Turn],
    resume_meta: list[str | None],
) -> tuple[int, list[str | None]]:
    """Sticky account for continuations; upload-capable account for files.

    Brand-new turns carrying uploads must land on an AVAILABLE session or
    the library gate rejects them. Resume metadata pins the account.
    """
    if resume_meta:
        state = await sessions.get(key)
        if (
            state is not None
            and state.account is not None
            and pool.has_slot(state.account)
        ):
            account = state.account
            return account, list(state.metadata) or resume_meta
        # Stale pin (pool resized): re-pin, then clear the foreign cid so the
        # new account starts fresh instead of resuming another session's cid.
        idx, _ = await _pick_account(pool, sessions, key)
        st, _ = await sessions.get_or_new(key)
        st.metadata = []
        await sessions.persist(key)
        return idx, []
    if _uploads_need_auth(turns):
        state = await sessions.get(key)
        if (
            state is not None
            and state.account is not None
            and pool.has_slot(state.account)
            and _account_is_upload_capable(pool, state.account)
        ):
            account = state.account
            return account, state.metadata
        picked = pool.pick_available()
        if picked is None:
            return await _pick_account(pool, sessions, key)
        st, _ = await sessions.get_or_new(key)
        st.account = picked[0]
        await sessions.persist(key)
        return picked[0], st.metadata
    return await _pick_account(pool, sessions, key)


async def _save_state(
    sessions: SessionStore,
    key: str,
    account: int,
    metadata: list[str | None],
) -> None:
    """Record the account and Gemini continuation metadata for a key."""
    state, _ = await sessions.get_or_new(key)
    state.account = account
    state.metadata = metadata
    await sessions.persist(key)


# ------------------------------------------------------- chat completions ---


class ChatPayload(NamedTuple):
    """Validated chat-completions request fields."""

    model: str | None
    stream: bool
    messages: JsonList
    conversation_id: str | None


async def _chat_payload(req: Request) -> ChatPayload:
    """Parse and validate a chat-completions request body."""
    body = await req.json()
    raw_messages = body.get("messages") or []
    messages = [m for m in raw_messages if isinstance(m, dict)]
    return ChatPayload(
        model=body.get("model") if isinstance(body.get("model"), str) else None,
        stream=bool(body.get("stream", False)),
        messages=messages,
        conversation_id=body.get("conversation_id")
        if isinstance(body.get("conversation_id"), str)
        else None,
    )


async def _resolve_chat_turns(
    http: httpx.AsyncClient,
    sessions: SessionStore,
    pool: AccountPool,
    messages: JsonList,
    conversation_id: str | None,
) -> ChatLocked:
    """Resolve the conversation key and new turns, validating trailing roles."""
    key = f"conv:{conversation_id}" if conversation_id else None
    if key is None:
        try:
            key, turns, resume_meta = await _chat_turns(
                http,
                sessions,
                pool,
                messages,
                conversation_id,
            )
        except ValueError as exc:
            msg = str(exc)
            raise ValueError(msg) from exc
        if not turns:
            msg = "no turns to send"
            raise ValueError(msg)
        return ChatLocked(key, turns, resume_meta, messages)
    if messages[-1].get("role", "user") in ("assistant", "tool"):
        # Stateless validation first: trailing assistant/tool carries nothing
        # new, regardless of whether server-side history exists yet.
        msg = "last message has role assistant/tool; nothing new to send"
        raise ValueError(msg)
    return ChatLocked(key, [], [], messages)


async def _handle_chat(req: Request) -> HandlerResult:
    """Handle a chat-completions request, streaming or buffered."""
    pool, sessions, http = _pool(req), _sessions(req), _http(req)
    try:
        payload = await _chat_payload(req)
    except (ValueError, AttributeError):
        return _err(400, "invalid JSON body", "invalid_request_error")
    model_name, stream, messages, conversation_id = payload
    if not messages:
        return _err(400, "messages is required", "invalid_request_error")
    model, thinking = _resolve_model(pool, model_name)
    # Resolve key first (cheap), then serialize everything account-touching.
    try:
        resolved = await _resolve_chat_turns(
            http, sessions, pool, messages, conversation_id
        )
    except ValueError as exc:
        return _err(400, str(exc), "invalid_request_error")
    convo_lock = await sessions.lock_for(resolved.key)
    async with convo_lock:
        key, turns, resume_meta = resolved.key, resolved.turns, resolved.resume_meta
        if conversation_id:
            try:
                key, turns, resume_meta = (
                    await _chat_turns(http, sessions, pool, messages, conversation_id)
                )[0:3]
            except ValueError as exc:
                return _err(400, str(exc), "invalid_request_error")
            if not turns:
                return _err(400, "no turns to send", "invalid_request_error")
        return await _handle_chat_locked(
            pool,
            sessions,
            http,
            ChatLocked(key, turns, resume_meta, messages),
            LockedModels(model_name, model, thinking, stream),
        )


class ChatLocked(NamedTuple):
    """Resolved chat conversation plus the turns to send through Gemini."""

    key: str
    turns: list[Turn]
    resume_meta: list[str | None]
    messages: JsonList


class ChatRunContext(NamedTuple):
    """Everything a chat run needs beyond resolved turns."""

    account: int
    metadata: list[str | None]
    model_name: str | None
    model: str | None
    prompt_for_usage: str
    thinking: bool


async def _buffered_chat(
    pool: AccountPool,
    sessions: SessionStore,
    http: httpx.AsyncClient,
    resolved: ChatLocked,
    ctx: ChatRunContext,
) -> HandlerResult:
    """Run turns and persist the completed chat-completion object."""
    key, turns, _, _messages = resolved
    try:
        out, new_meta = await _run_turns(
            pool,
            RunRequest(ctx.account, ctx.model, ctx.thinking, turns, ctx.metadata),
        )
    except (
        GeminiError,
        AuthError,
        APIError,
        httpx.HTTPError,
        OSError,
        ValueError,
    ) as exc:
        await sessions.drop_if_fresh(key)
        code, msg = _map_exception(exc)
        return _err(code, msg)
    md = await _gemini_images_to_markdown(http, out)
    text = _full_text(out) + md
    thoughts = _thoughts(out)
    await _save_state(sessions, key, ctx.account, new_meta)
    return _chat_completion(ctx.model_name, key, text, thoughts, ctx.prompt_for_usage)


async def _stream_chat(
    pool: AccountPool,
    sessions: SessionStore,
    http: httpx.AsyncClient,
    resolved: ChatLocked,
    ctx: ChatRunContext,
) -> AsyncIterator[str]:
    """Stream chat deltas, persisting completion state on the done event."""
    key, turns, _, _ = resolved
    cid = _new_id("chatcmpl-")
    state: Json = {"sent_role": False, "full": ""}
    try:
        async for kind, payload in _run_turns_stream(
            pool,
            RunRequest(
                ctx.account,
                ctx.model,
                ctx.thinking,
                turns,
                ctx.metadata,
            ),
        ):
            if kind == "delta" and isinstance(payload, str):
                async for frame in _chat_delta_frames(
                    cid,
                    ctx.model_name,
                    payload,
                    state,
                ):
                    yield frame
            elif kind == "done" and isinstance(payload, tuple):
                out, new_meta = payload
                if not isinstance(new_meta, list):
                    continue
                if not isinstance(out, ModelOutput):
                    await _save_state(sessions, key, ctx.account, new_meta)
                    continue
                async for frame in _chat_done_frames(
                    http,
                    sessions,
                    ChatDoneContext(
                        cid,
                        key,
                        ctx.account,
                        new_meta,
                        ctx.model_name,
                        ctx.prompt_for_usage,
                        out,
                    ),
                    state,
                ):
                    yield frame
    except (
        GeminiError,
        AuthError,
        APIError,
        httpx.HTTPError,
        OSError,
    ) as exc:
        await sessions.drop_if_fresh(key)
        yield _sse({"error": {"message": str(exc) or type(exc).__name__}})
    yield "data: [DONE]\n\n"


class LockedModels(NamedTuple):
    """Model selection for a locked conversation run."""

    model_name: str | None
    model: str | None
    thinking: bool
    stream: bool


async def _handle_chat_locked(
    pool: AccountPool,
    sessions: SessionStore,
    http: httpx.AsyncClient,
    resolved: ChatLocked,
    models: LockedModels,
) -> HandlerResult:
    key, turns, resume_meta, _ = resolved
    account, metadata = await _pick_account_for_turns(
        pool,
        sessions,
        key,
        turns,
        resume_meta,
    )
    prompt_for_usage = "\n".join(t for t, _ in turns)
    ctx = ChatRunContext(
        account,
        metadata,
        models.model_name,
        models.model,
        prompt_for_usage,
        models.thinking,
    )
    if not models.stream:
        return await _buffered_chat(pool, sessions, http, resolved, ctx)

    async def gen() -> AsyncIterator[str]:
        async for frame in _stream_chat(pool, sessions, http, resolved, ctx):
            yield frame

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(obj: Json) -> str:
    """Encode one server-sent-events data frame."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions", response_model=None)
@app.post("/chat/completions", response_model=None)
async def chat_completions(req: Request) -> HandlerResult:
    """Create a chat completion; JSON errors stay JSON, never exceptions."""
    try:
        return await _handle_chat(req)
    except (ValueError, AttributeError, KeyError):
        return _err(400, "invalid JSON body", "invalid_request_error")


# ------------------------------------------------------------- responses ---


def _norm_input_item(item: Json | str | float | None) -> str:
    """Normalize one Responses input item for prefix-overlap comparison."""
    if not isinstance(item, dict):
        return json.dumps(item, sort_keys=True, default=str)
    itype = item.get("type", "message")
    if itype != "message":
        return json.dumps(
            {k: item.get(k) for k in ("type", "id")},
            sort_keys=True,
            default=str,
        )
    parts: list[str] = []
    content = item.get("content", [])
    if isinstance(content, str):
        parts.append(content)
    else:
        for p in content or []:
            if not isinstance(p, dict):
                continue
            if p.get("type") in ("input_text", "output_text", "text"):
                text = p.get("text", "")
                parts.append(text if isinstance(text, str) else "")
            else:
                parts.append(part_identity(p))
    return f"{item.get('role', '')}:{('|'.join(parts))}"


def _prefix_overlap(prev_norm: list[str], new_norm: list[str]) -> int:
    """Count the shared leading items between stored and incoming norms."""
    cut = 0
    for a, b in zip(prev_norm, new_norm, strict=False):
        if a != b:
            break
        cut += 1
    return cut


async def _items_to_turns(
    http: httpx.AsyncClient,
    items: JsonList,
    *,
    skip_roles: tuple[str, ...] = ("assistant", "tool"),
) -> list[Turn]:
    """Convert Responses items to user turns, dropping echoed roles."""
    turns: list[Turn] = []
    for it in items:
        role, text, uploads = await message_to_turn(
            http,
            {"role": it.get("role", "user"), "content": it.get("content", "")},
        )
        if role in skip_roles:
            for up, _ in uploads:
                up.cleanup()
            continue
        turns.append((text, uploads))
    return turns


def _with_instructions(turns: list[Turn], instructions: str | None) -> list[Turn]:
    """Prepend system instructions to a turn list when present."""
    if instructions:
        return [(f"[System] {instructions}", []), *turns]
    return turns


class ResponseTurns(NamedTuple):
    """Inputs for resolving Responses turns against stored history."""

    items: JsonList
    instructions: str | None
    prev_response_id: str | None
    conversation: str | None


async def _fresh_response_turns(
    http: httpx.AsyncClient,
    items: JsonList,
    instructions: str | None,
) -> list[Turn]:
    """Build turns for a brand-new Responses conversation key."""
    return _with_instructions(
        await _items_to_turns(http, items, skip_roles=("assistant",)),
        instructions,
    ) or [(" ", [])]


async def _responses_turns(
    http: httpx.AsyncClient,
    sessions: SessionStore,
    request: ResponseTurns,
) -> tuple[str, list[Turn], list[str | None], str | None]:
    """Resolve (state_key, NEW turns only, resume metadata, prev text) for Responses.

    Resumed server-side history is never replayed: only trailing items not
    already covered by the stored input norm are sent.
    """
    items, instructions, prev_response_id, conversation = request
    if conversation:
        key = f"conv:{conversation}"
        state, created = await sessions.get_or_new(key)
        if not created and state.metadata:
            prev_norm: list[str] = (await sessions.get_norm(key)) or []
            cut = _prefix_overlap(prev_norm, [_norm_input_item(i) for i in items])
            turns = _with_instructions(
                await _items_to_turns(http, items[cut:]),
                instructions,
            )
            return key, turns, state.metadata, None
        return key, await _fresh_response_turns(http, items, instructions), [], None

    if prev_response_id:
        saved = await sessions.get_response(prev_response_id)
        if saved and isinstance(saved.get("_state"), str):
            key = saved["_state"]
            state = await sessions.get(key)
            old_norm = saved.get("_norm") or []
            old_list = old_norm if isinstance(old_norm, list) else []
            cut = _prefix_overlap(old_list, [_norm_input_item(i) for i in items])
            turns = _with_instructions(
                await _items_to_turns(http, items[cut:]),
                instructions,
            )
            prev_text = saved.get("_text")
            return (
                key,
                turns,
                (state.metadata if state else []),
                prev_text if isinstance(prev_text, str) else None,
            )

    key = _response_key(items)
    await sessions.get_or_new(key)
    return key, await _fresh_response_turns(http, items, instructions), [], None


def _response_key(items: JsonList) -> str:
    """Build a fresh Responses conversation key from normalized input."""
    digest = hashlib.sha256(
        "|".join(_norm_input_item(i) for i in items).encode(),
    ).hexdigest()[:12]
    return f"resp:{uuid.uuid4().hex[:16]}:{digest}"


def _prompt_response_key(prompt: str) -> str:
    """Build a fresh Responses conversation key from a raw string prompt."""
    digest = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    return f"resp:{uuid.uuid4().hex[:16]}:{digest}"


class ResponseContent(NamedTuple):
    """Fields for building one Responses object."""

    rid: str
    model_name: str | None
    text: str
    conv_id: str
    prev_id: str | None
    prompt: str
    key: str
    items: JsonList


def _response_object(
    content: ResponseContent,
    status: str = "completed",
) -> dict[str, Any]:
    """Build the public Responses object for one completed turn."""
    return {
        "id": content.rid,
        "object": "response",
        "created_at": _now(),
        "model": content.model_name or "gemini-flash",
        "status": status,
        "output": [
            {
                "type": "message",
                "id": _new_id("msg_"),
                "status": status,
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": content.text, "annotations": []},
                ],
            },
        ],
        "usage": {
            "input_tokens": max(1, len(content.prompt) // 4),
            "output_tokens": max(1, len(content.text) // 4),
            "total_tokens": max(2, (len(content.prompt) + len(content.text)) // 4),
        },
        "conversation": {"id": content.conv_id},
        "previous_response_id": content.prev_id,
    }


def _save_response_obj(content: ResponseContent) -> dict[str, Any]:
    """Build a response object plus its private chaining fields."""
    obj = _response_object(content)
    obj["_state"] = content.key
    obj["_text"] = content.text
    obj["_norm"] = [_norm_input_item(i) for i in content.items]
    return obj


def _response_event(
    event: str,
    response: dict[str, Any],
    model_name: str | None,
) -> str:
    """Encode a Responses lifecycle event (created) as an SSE frame."""
    body = {
        "type": event,
        "response": {**response, "model": model_name or "gemini-flash"},
    }
    return f"event: {event}\ndata: {json.dumps(body)}\n\n"


def _delta_event(rid: str, delta: str) -> str:
    """Encode a Responses output-text delta as an SSE frame."""
    body = {
        "type": "response.output_text.delta",
        "item_id": rid,
        "delta": delta,
    }
    return f"event: response.output_text.delta\ndata: {json.dumps(body)}\n\n"


def _completed_event(response: dict[str, Any]) -> str:
    """Encode a completed Responses object as an SSE frame."""
    body = {"type": "response.completed", "response": response}
    return f"event: response.completed\ndata: {json.dumps(body)}\n\n"


def _failed_event(exc: BaseException) -> str:
    """Encode a Responses failure as an SSE frame."""
    body = {"type": "response.failed", "error": str(exc)}
    return f"event: response.failed\ndata: {json.dumps(body)}\n\n"


class ResponsesPayload(NamedTuple):
    """Validated Responses request fields."""

    model: str | None
    raw_input: str | list[Json]
    instructions: str | None
    prev_id: str | None
    conv_id: str | None
    stream: bool
    thinking: bool
    resolved_model: str | None


async def _responses_payload(req: Request, pool: AccountPool) -> ResponsesPayload:
    """Parse and validate a Responses request body."""
    body = await req.json()
    raw_input = body.get("input", "")
    model = body.get("model") if isinstance(body.get("model"), str) else None
    instructions = body.get("instructions")
    prev_id = body.get("previous_response_id")
    conv_param = body.get("conversation")
    conv_id = conv_param.get("id") if isinstance(conv_param, dict) else conv_param
    resolved, thinking = _resolve_model(pool, model)
    return ResponsesPayload(
        model=model,
        raw_input=raw_input if isinstance(raw_input, str | list) else "",
        instructions=instructions if isinstance(instructions, str) else None,
        prev_id=prev_id if isinstance(prev_id, str) else None,
        conv_id=conv_id if isinstance(conv_id, str) else None,
        stream=bool(body.get("stream", False)),
        thinking=thinking,
        resolved_model=resolved,
    )


def _response_items(raw_input: str | list[Json]) -> list[dict[str, Any]]:
    """Coerce Responses input to a list of message dicts."""
    if isinstance(raw_input, str):
        return []
    return [i for i in raw_input if isinstance(i, dict)]


class ResolvedResponses(NamedTuple):
    """Validated Responses inputs plus the resolved conversation key."""

    items: JsonList
    key: str | None
    pre_turns: list[Turn] | None
    turns: list[Turn]
    resume_meta: list[str | None]


def _prompt_items(prompt: str) -> JsonList:
    """Wrap a raw string prompt as a single Responses user message."""
    if not prompt.strip():
        return []
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        },
    ]


async def _resolve_responses_inputs(
    sessions: SessionStore,
    payload: ResponsesPayload,
) -> ResolvedResponses:
    """Resolve Responses payload to items, key, and pre-built string turns."""
    raw_input = payload.raw_input
    if isinstance(raw_input, str):
        prompt = raw_input
        if payload.prev_id or payload.conv_id:
            return ResolvedResponses(_prompt_items(prompt), None, None, [], [])
        pre_key = _prompt_response_key(prompt)
        pre_turns = (
            [(f"[System] {payload.instructions}\n\n{prompt}", [])]
            if payload.instructions
            else [(prompt, [])]
        )
        await sessions.get_or_new(pre_key)
        return ResolvedResponses([], pre_key, pre_turns, pre_turns, [])
    return ResolvedResponses(_response_items(raw_input), None, None, [], [])


async def _resolve_responses_key(
    sessions: SessionStore,
    payload: ResponsesPayload,
    resolved: ResolvedResponses,
) -> str:
    """Peek the conversation key without downloading files."""
    if resolved.key is not None:
        return resolved.key
    if not resolved.items and resolved.pre_turns is None:
        msg = "no input items to send"
        raise ValueError(msg)
    if payload.conv_id:
        return f"conv:{payload.conv_id}"
    if payload.prev_id:
        saved = await sessions.get_response(payload.prev_id)
        state_key = saved.get("_state") if saved else None
        return state_key if isinstance(state_key, str) else f"pending:{payload.prev_id}"
    return _response_key(resolved.items)


async def _handle_responses(req: Request) -> HandlerResult:
    """Handle a Responses request, streaming or buffered."""
    pool, sessions, http = _pool(req), _sessions(req), _http(req)
    try:
        payload = await _responses_payload(req, pool)
    except (ValueError, AttributeError):
        return _err(400, "invalid JSON body", "invalid_request_error")
    try:
        resolved = await _resolve_responses_inputs(sessions, payload)
        key = await _resolve_responses_key(sessions, payload, resolved)
    except ValueError as exc:
        return _err(400, str(exc), "invalid_request_error")
    items, pre_turns = resolved.items, resolved.pre_turns
    turns, resume_meta = resolved.turns, resolved.resume_meta
    peek_key = key
    convo_lock = await sessions.lock_for(peek_key)
    real_key: str | None = None
    async with convo_lock:
        if pre_turns is None:
            try:
                real_key, turns, resume_meta, _ = await _responses_turns(
                    http,
                    sessions,
                    ResponseTurns(
                        items,
                        payload.instructions,
                        payload.prev_id,
                        payload.conv_id,
                    ),
                )
            except ValueError as exc:
                return _err(400, str(exc), "invalid_request_error")
            key = real_key
        if not turns:
            return _err(400, "no input items to send", "invalid_request_error")
        result = await _handle_responses_locked(
            pool,
            sessions,
            http,
            LockedTurns(key, turns, resume_meta, items),
            LockedResponseModels(
                payload.model,
                payload.resolved_model,
                payload.stream,
                payload.prev_id,
                payload.thinking,
            ),
        )
    # _responses_turns mints a fresh key for new conversations; the peeked
    # random key is then junk: drop it while fresh. Outside the convo lock
    # because drop takes the store guard (lock_for already released here).
    if real_key is not None and real_key != peek_key:
        await sessions.drop_if_fresh(peek_key)
    return result


class LockedTurns(NamedTuple):
    """Resolved conversation key plus the turns to send through Gemini."""

    key: str
    turns: list[Turn]
    resume_meta: list[str | None]
    items: JsonList


class BufferedContext(NamedTuple):
    """Everything a buffered Responses run needs beyond resolved turns."""

    account: int
    metadata: list[str | None]
    model_name: str | None
    model: str | None
    conv_out: str
    prev_id: str | None
    prompt_for_usage: str
    thinking: bool


async def _buffered_response(
    pool: AccountPool,
    sessions: SessionStore,
    http: httpx.AsyncClient,
    resolved: LockedTurns,
    ctx: BufferedContext,
) -> HandlerResult:
    """Run turns and persist the completed Responses object."""
    key, turns, _, items = resolved
    rid = _new_id("resp_")
    try:
        out, new_meta = await _run_turns(
            pool,
            RunRequest(ctx.account, ctx.model, ctx.thinking, turns, ctx.metadata),
        )
    except (
        GeminiError,
        AuthError,
        APIError,
        httpx.HTTPError,
        OSError,
        ValueError,
    ) as exc:
        await sessions.drop_if_fresh(key)
        code, msg = _map_exception(exc)
        return _err(code, msg)
    md = await _gemini_images_to_markdown(http, out)
    text = _full_text(out) + md
    await _save_state(sessions, key, ctx.account, new_meta)
    await sessions.set_norm(key, [_norm_input_item(i) for i in items])
    obj = _save_response_obj(
        ResponseContent(
            rid,
            ctx.model_name,
            text,
            ctx.conv_out,
            ctx.prev_id,
            ctx.prompt_for_usage,
            key,
            items,
        ),
    )
    await sessions.save_response(rid, obj)
    return {k: v for k, v in obj.items() if not k.startswith("_")}


class LockedResponseModels(NamedTuple):
    """Model selection for a locked Responses run."""

    model_name: str | None
    model: str | None
    stream: bool
    prev_id: str | None
    thinking: bool


async def _handle_responses_locked(
    pool: AccountPool,
    sessions: SessionStore,
    http: httpx.AsyncClient,
    resolved: LockedTurns,
    models: LockedResponseModels,
) -> HandlerResult:
    key, turns, resume_meta, items = resolved
    account, metadata = await _pick_account_for_turns(
        pool,
        sessions,
        key,
        turns,
        resume_meta,
    )
    prompt_for_usage = "\n".join(t for t, _ in turns)
    conv_out = key.removeprefix("conv:")
    if not models.stream:
        return await _buffered_response(
            pool,
            sessions,
            http,
            resolved,
            BufferedContext(
                account,
                metadata,
                models.model_name,
                models.model,
                conv_out,
                models.prev_id,
                prompt_for_usage,
                models.thinking,
            ),
        )

    async def gen() -> AsyncIterator[str]:
        rid = _new_id("resp_")
        yield _response_event(
            "response.created",
            {"id": rid, "object": "response", "status": "in_progress"},
            models.model_name,
        )

        full = ""
        try:
            async for kind, payload in _run_turns_stream(
                pool,
                RunRequest(
                    account,
                    models.model,
                    models.thinking,
                    turns,
                    metadata,
                ),
            ):
                if kind == "delta" and isinstance(payload, str):
                    full += payload
                    yield _delta_event(rid, payload)
                elif kind == "done" and isinstance(payload, tuple):
                    out, new_meta = payload
                    if not isinstance(new_meta, list):
                        continue
                    md = (
                        await _gemini_images_to_markdown(http, out)
                        if isinstance(
                            out,
                            ModelOutput,
                        )
                        else ""
                    )
                    if md:
                        full += md
                        yield _delta_event(rid, md)
                    await _save_state(sessions, key, account, new_meta)
                    await sessions.set_norm(key, [_norm_input_item(i) for i in items])
                    obj = _save_response_obj(
                        ResponseContent(
                            rid,
                            models.model_name,
                            full,
                            conv_out,
                            models.prev_id,
                            prompt_for_usage,
                            key,
                            items,
                        ),
                    )
                    await sessions.save_response(rid, obj)
                    public = {k: v for k, v in obj.items() if not k.startswith("_")}
                    yield _completed_event(public)
        except (
            GeminiError,
            AuthError,
            APIError,
            httpx.HTTPError,
            OSError,
        ) as exc:
            await sessions.drop_if_fresh(key)
            yield _failed_event(exc)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/responses", response_model=None)
@app.post("/responses", response_model=None)
async def create_response(req: Request) -> HandlerResult:
    """Create a response; JSON errors stay JSON, never exceptions."""
    try:
        return await _handle_responses(req)
    except (ValueError, AttributeError, KeyError):
        return _err(400, "invalid JSON body", "invalid_request_error")


@app.get("/v1/responses/{rid}", response_model=None)
@app.get("/responses/{rid}", response_model=None)
async def get_response(req: Request, rid: str) -> HandlerResult:
    """Fetch one saved response object by id."""
    saved = await _sessions(req).get_response(rid)
    if not saved:
        return _err(404, f"response {rid} not found", "invalid_request_error")
    return {k: v for k, v in saved.items() if not k.startswith("_")}


# ----------------------------------------------------------------- models ---


@app.get("/v1/models", response_model=None)
@app.get("/models", response_model=None)
async def list_models(req: Request) -> Json:
    """List registry models plus display-slug and flash aliases."""
    return {"object": "list", "data": _model_list(_pool(req))}


# ------------------------------------------------------------------ files ---


@app.post("/v1/files", response_model=None)
async def upload_file(req: Request) -> HandlerResult:
    """Store an uploaded file; bytes persist in SQLite across restarts."""
    form = await req.form()
    upload = form.get("file")
    purpose = form.get("purpose")
    if not isinstance(upload, StarletteUploadFile):
        return _err(400, "file is required (multipart)", "invalid_request_error")
    data = await upload.read()
    if not isinstance(data, bytes):
        return _err(400, "file is required (multipart)", "invalid_request_error")
    filename = upload.filename or "upload.bin"
    purpose_str = purpose if isinstance(purpose, str) else "assistants"

    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    stored_purpose = purpose_str or "assistants"
    try:
        fid = await UploadStore.aput(data, mime, filename, stored_purpose)
    except ValueError as exc:
        return _err(413, str(exc), "invalid_request_error")
    return {
        "id": fid,
        "object": "file",
        "bytes": len(data),
        "created_at": _now(),
        "filename": filename,
        "purpose": stored_purpose,
    }


@app.get("/v1/files", response_model=None)
async def list_files() -> Json:
    """List uploaded files, oldest first."""
    return {
        "object": "list",
        "data": [
            {
                "id": fid,
                "object": "file",
                "bytes": size,
                "created_at": created,
                "filename": fn,
                "purpose": purpose,
            }
            for fid, size, fn, purpose, created in await UploadStore.alist_all()
        ],
    }


@app.get("/v1/files/{fid}", response_model=None)
async def retrieve_file(fid: str) -> HandlerResult:
    """Fetch one uploaded file's metadata by id."""
    stored = await UploadStore.aget(fid)
    if not stored:
        return _err(404, f"file {fid} not found", "invalid_request_error")
    data, _, filename, purpose, created = stored
    return {
        "id": fid,
        "object": "file",
        "bytes": len(data),
        "created_at": created,
        "filename": filename,
        "purpose": purpose,
    }


@app.delete("/v1/files/{fid}", response_model=None)
async def delete_file(fid: str) -> HandlerResult:
    """Delete one uploaded file by id."""
    if not await UploadStore.adelete(fid):
        return _err(404, f"file {fid} not found", "invalid_request_error")
    return {"id": fid, "object": "file", "deleted": True}


@app.get("/v1/files/{fid}/content", response_model=None)
async def file_content(fid: str) -> Response:
    """Download one uploaded file's bytes by id."""
    stored = await UploadStore.aget(fid)
    if not stored:
        return _err(404, f"file {fid} not found", "invalid_request_error")
    data, mime, _, _, _ = stored
    return Response(content=data, media_type=mime)


# ----------------------------------------------------------------- images ---


async def _rehost_generated_image(
    http: httpx.AsyncClient,
    img: GeneratedImage | Image,
    prompt: str,
    idx: int,
) -> dict[str, str] | None:
    """Save one generated image and re-host it; fall back to the Gemini URL."""
    url = (img.url or "").strip()
    try:
        path = await img.save(
            path=str(IMG_TMP),
            filename=f"gen_{uuid.uuid4().hex[:8]}",
        )
    except (OSError, ValueError, RuntimeError, httpx.HTTPError):
        return {"url": url, "revised_prompt": prompt} if url else None
    try:
        raw = await asyncio.to_thread(Path(path).read_bytes)
    except (OSError, ValueError, RuntimeError):
        return {"url": url, "revised_prompt": prompt} if url else None
    finally:
        await asyncio.to_thread(Path(path).unlink, missing_ok=True)
    public = (await freeimage.upload_png(http, raw, f"gemini_{idx}.png") or "").strip()
    final = public or url
    return {"url": final, "revised_prompt": prompt} if final else None


async def _handle_image_gen(req: Request) -> HandlerResult:
    """Generate images for a prompt and return re-hosted URLs."""
    pool, http = _pool(req), _http(req)
    try:
        body = await req.json()
    except (ValueError, AttributeError):
        return _err(400, "invalid JSON body", "invalid_request_error")
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        return _err(400, "prompt is required", "invalid_request_error")
    model_name = body.get("model") if isinstance(body.get("model"), str) else None
    model, thinking = _resolve_model(pool, model_name)
    # Image gen is single-shot: plain round-robin, no durable session row.
    account, _ = pool.pick()
    try:
        out, _ = await _run_turns(
            pool,
            RunRequest(account, model, thinking, [(prompt, [])], []),
        )
    except (
        GeminiError,
        AuthError,
        APIError,
        httpx.HTTPError,
        OSError,
        ValueError,
    ) as exc:
        code, msg = _map_exception(exc)
        return _err(code, msg)
    imgs = list(out.images or [])
    if not imgs:
        return _err(500, "Gemini returned no image for this prompt")
    data = [
        entry
        for entry in await asyncio.gather(
            *(
                _rehost_generated_image(http, img, prompt, i)
                for i, img in enumerate(imgs)
            ),
        )
        if entry
    ]
    if not data:
        return _err(500, "failed to retrieve generated image bytes")
    return {"created": _now(), "data": data}


@app.post("/v1/images/generations", response_model=None)
@app.post("/images/generations", response_model=None)
async def image_generations(req: Request) -> HandlerResult:
    """Generate images; JSON errors stay JSON, never exceptions."""
    try:
        return await _handle_image_gen(req)
    except (ValueError, AttributeError, KeyError):
        return _err(400, "invalid JSON body", "invalid_request_error")


# ------------------------------------------------------------------ misc ---


_COOKIE_SYNC_INTERVAL = 600.0


async def _cookie_sync_loop(pool: AccountPool) -> None:
    """Refresh accounts.txt from Chrome; heal degraded clients; never raises."""
    try:
        while True:
            await asyncio.sleep(_COOKIE_SYNC_INTERVAL)
            try:
                chrome_synced = chrome_cookies.refresh_accounts_from_chrome(
                    ACCOUNTS_FILE,
                )
            except Exception as exc:  # noqa: BLE001 - loop must never die silently
                _log.warning("gem2oai: periodic Chrome sync skipped: %s", exc)
                continue
            try:
                healed = await _heal_degraded_accounts(pool)
            except Exception as exc:  # noqa: BLE001 - loop must never die silently
                _log.warning("gem2oai: account heal skipped: %s", exc)
                healed = 0
            try:
                live, expiries, attrs = pool_live_state(pool)
                synced = sync_accounts_file(
                    ACCOUNTS_FILE,
                    live,
                    expiries,
                    attrs,
                )
            except Exception as exc:  # noqa: BLE001 - loop must never die silently
                _log.warning("gem2oai: periodic cookie sync skipped: %s", exc)
                continue
            if chrome_synced or healed or synced:
                _log.info(
                    "gem2oai: cookie sync: chrome=%d healed=%d live=%d",
                    chrome_synced,
                    healed,
                    synced,
                )
    except asyncio.CancelledError:
        pass


async def _heal_degraded_accounts(pool: AccountPool) -> int:
    """Reinit non-AVAILABLE accounts from live Chrome cookies.

    Chrome is the source of truth: when Google invalidates a session the
    browser picks up fresh cookies on next use, so a degraded account
    heals itself here without a restart. Replacement clients init outside
    the per-account lock (pinned turns keep flowing on the old client);
    the lock covers only the AVAILABLE re-check plus pointer swap.
    Returns accounts healed.
    """
    import sqlite3

    dirs = chrome_cookies.profile_dirs()
    healed = 0
    for index, profile_dir in enumerate(dirs):
        if index >= pool.client_count():
            break
        try:
            client = pool.client_at(index)
        except (IndexError, TypeError, ValueError):
            continue
        if client.account_status == AccountStatus.AVAILABLE:
            continue
        try:
            fresh_cookies = chrome_cookies.jar_dict(profile_dir)
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
            _log.warning("gem2oai: heal skipped for account %d: %s", index, exc)
            continue
        if not fresh_cookies.get("__Secure-1PSID"):
            continue
        try:
            fresh = await pool.build_replacement(index, fresh_cookies)
        except (AttributeError, TypeError, ValueError, OSError) as exc:
            _log.warning(
                "gem2oai: heal failed for account %d: %s",
                index,
                exc,
            )
            continue
        if fresh is None:
            pool.report(index, ok=False)
            continue
        try:
            lock = pool.lock_for(index)
        except (IndexError, TypeError, ValueError):
            with suppress(Exception):
                await fresh.close()
            continue
        async with lock:
            try:
                if pool.client_at(index).account_status == AccountStatus.AVAILABLE:
                    keep = False
                else:
                    pool.swap_client(index, fresh)
                    keep = True
            except (IndexError, TypeError, ValueError) as exc:
                _log.warning(
                    "gem2oai: heal failed for account %d: %s",
                    index,
                    exc,
                )
                keep = False
        if keep:
            healed += 1
            _log.info("gem2oai: healed account %d from Chrome", index)
        else:
            with suppress(Exception):
                await fresh.close()
    return healed


def _redis(req: Request) -> RedisState | None:
    """Return the shared Redis handle, or None when SQLite-only."""
    return getattr(req.app.state, "redis", None)


async def _redis_health(req: Request) -> Json:
    """Probe Redis liveness plus hit-ratio/throughput signals for /health."""
    redis_state = _redis(req)
    if redis_state is None:
        return {"configured": False, "status": "disabled"}
    try:
        latency = await redis_state.ping_ms()
        stats = await redis_state.stats()
    except Exception as exc:  # noqa: BLE001 - health must report, never raise
        return {"configured": True, "status": "down", "error": str(exc)}
    return {
        "configured": True,
        "status": "ok",
        "latency_ms": round(latency, 2),
        **stats,
    }


@app.get("/health", response_model=None)
@app.get("/", response_model=None)
async def health(req: Request) -> Json:
    """Report liveness, auth state, account count, freeimage key presence."""
    try:
        pool = _pool(req)
    except (AttributeError, TypeError, ValueError):
        return {
            "status": "ok",
            "accounts": 0,
            "auth": "unknown",
            "freeimage": bool(freeimage_api_key()),
            "redis": await _redis_health(req),
        }
    try:
        state = pool.auth_state()
    except (AttributeError, TypeError, ValueError):
        state = "unknown"
    return {
        "status": "ok",
        "accounts": pool.client_count(),
        "auth": state,
        "freeimage": bool(freeimage_api_key()),
        "redis": await _redis_health(req),
    }


@app.get("/v1/conversations/{cid}", response_model=None)
async def get_conversation(req: Request, cid: str) -> HandlerResult:
    """Report whether a conversation id has resumable Gemini history."""
    state = await _sessions(req).get(f"conv:{cid}")
    if state is None:
        return _err(404, f"conversation {cid} not found", "invalid_request_error")
    return {"id": cid, "object": "conversation", "has_history": bool(state.metadata)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
