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
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import freeimage
from accounts import AccountPool, load_account_cookies
from config import ACCOUNTS_FILE, PORT, freeimage_api_key
from content import (
    UploadStore,
    message_to_turn,
    openai_messages_to_prompt,
    part_identity,
    responses_input_to_prompt,
)
from conversations import SessionStore

IMG_TMP = Path(tempfile.gettempdir()) / "gem2oai-imgs"
IMG_TMP.mkdir(parents=True, exist_ok=True)

THINKING_ALIASES = {
    "gemini-3.8-flash-thinking",
    "gemini-3.8-flash-extended-thinking",
    "gemini-3.8-flash-et",
}


def _now() -> int:
    return int(time.time())


def _new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------- models ---


def _resolve_model(pool: AccountPool, name: str | None) -> tuple[str | None, bool]:
    """OpenAI model name -> (gemini registry name, extended_thinking)."""
    n = (name or "").strip().lower()
    thinking = "think" in n or n.endswith("-et")
    if not n or n in ("default", "auto"):
        return None, thinking
    if n in THINKING_ALIASES:
        return "gemini-flash", True
    core = n.replace("extended", "").replace("thinking", "").replace("-et", "")
    if "lite" in core or "3.5" in core:
        return "gemini-flash-lite", thinking
    if "pro" in core or "3.1" in core:
        return "gemini-pro", thinking
    if "flash" in core or "3.8" in core or "3.6" in core:
        return "gemini-flash", thinking
    return pool.resolve(name), thinking


def _model_list(pool: AccountPool) -> list[dict]:
    seen: dict[str, dict] = {m["id"]: m for m in pool.models()}
    for slug in pool.display_slugs():
        seen.setdefault(
            slug, {"id": slug, "object": "model", "created": 0, "owned_by": "gemini"}
        )
    for alias in ["gemini-3.8-flash", *sorted(THINKING_ALIASES)]:
        seen.setdefault(
            alias, {"id": alias, "object": "model", "created": 0, "owned_by": "gemini"}
        )
    return [seen[k] for k in sorted(seen)]


# --------------------------------------------------------------- app boot ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    cookies = load_account_cookies(str(ACCOUNTS_FILE))
    pool = AccountPool(cookies)
    try:
        await pool.init_all()
    except RuntimeError as exc:
        print(f"gem2oai: FATAL: {exc}")
        raise RuntimeError(str(exc)) from exc
    app.state.pool = pool
    app.state.sessions = SessionStore()
    app.state.http = httpx.AsyncClient(timeout=120, follow_redirects=True)
    print(f"gem2oai: {len(cookies)} account(s) ready, freeimage={'yes' if freeimage_api_key() else 'NO KEY'}")
    yield
    await app.state.http.aclose()
    await pool.close_all()

app = FastAPI(title="gemini.google.com-to-openai-api", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
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


def _is_upstream_error(exc: Exception) -> bool:
    """True when the failure implicates the Gemini account/network, not our input."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if any(k in name for k in ("Timeout", "Auth", "UsageLimit", "Blocked", "APIError", "GeminiError")):
        return True
    return any(
        k in msg
        for k in ("timed out", "timeout", "429", "503", "500", "unauth", "auth", "quota", "blocked", "network", "connection", "reset")
    )


def _map_exception(exc: Exception) -> tuple[int, str]:
    name = type(exc).__name__
    msg = str(exc) or name
    if "UsageLimitExceeded" in name or "429" in msg:
        return 429, msg
    if "Timeout" in name or "timed out" in msg.lower():
        return 504, msg
    if "Auth" in name or "unauth" in msg.lower():
        return 502, msg
    return 500, msg

# ------------------------------------------------------------ gemini run ---


async def _run_turns(
    pool: AccountPool,
    account: int,
    model: str | None,
    thinking: bool,
    turns: list[tuple[str, list]],
    metadata: list,
) -> tuple[Any, list]:
    """Replay turns through one native ChatSession. Returns (final output, metadata)."""
    client = _client_at(pool, account) if account >= 0 else pool.pick()[1]
    chat = client.start_chat(
        model=model, **({"metadata": metadata} if metadata else {})
    )
    out = None
    try:
        async with pool.lock_for(account):
            for text, uploads in turns:
                files = [str(up.path) for up, _ in uploads] or None
                try:
                    out = await chat.send_message(
                        text.strip() or " ",
                        files=files,
                        extended_thinking=thinking,
                    )
                finally:
                    for up, _ in uploads:
                        up.cleanup()
        pool.report(account, True)
        return out, list(chat.metadata)
    except Exception as exc:
        pool.report(account, not _is_upstream_error(exc))
        raise


def _client_at(pool: AccountPool, account: int):
    return pool._entries[account].client




async def _run_turns_stream(
    pool: AccountPool,
    account: int,
    model: str | None,
    thinking: bool,
    turns: list[tuple[str, list]],
    metadata: list,
):
    """Replay prior turns, then stream the final turn's deltas.

    Yields ("delta", text) chunks, then ("done", (output, metadata)).
    Holds the account lock for the whole conversation update.
    Raises ValueError when turns is empty (caller must 400, not stream).
    """
    if not turns:
        raise ValueError("no turns to send")
    client = _client_at(pool, account)
    chat = client.start_chat(
        model=model, **({"metadata": metadata} if metadata else {})
    )
    lock = pool.lock_for(account)
    await lock.acquire()
    try:
        out = None
        for text, uploads in turns[:-1]:
            files = [str(up.path) for up, _ in uploads] or None
            try:
                out = await chat.send_message(
                    text.strip() or " ",
                    files=files,
                    extended_thinking=thinking,
                )
            finally:
                for up, _ in uploads:
                    up.cleanup()
        text, uploads = turns[-1]
        files = [str(up.path) for up, _ in uploads] or None
        try:
            async for chunk in chat.send_message_stream(
                text.strip() or " ",
                files=files,
                extended_thinking=thinking,
            ):
                out = chunk
                if chunk.text_delta:
                    yield ("delta", chunk.text_delta)
        finally:
            for up, _ in uploads:
                up.cleanup()
        pool.report(account, True)
        yield ("done", (out, list(chat.metadata)))
    except Exception as exc:
        pool.report(account, not _is_upstream_error(exc))
        raise
    finally:
        lock.release()


# ---------------------------------------------------------------- images ---


async def _gemini_images_to_markdown(
    http: httpx.AsyncClient, out: Any
) -> str:
    """Download generated/web images, re-host on freeimage, return markdown block."""
    imgs = list(getattr(out, "images", None) or [])
    if not imgs:
        return ""
    md: list[str] = []

    async def one(idx: int, img: Any) -> str | None:
        url = getattr(img, "url", "")
        try:
            path = await img.save(path=str(IMG_TMP), filename=f"gen_{uuid.uuid4().hex[:8]}")
            data = Path(path).read_bytes()
            try:
                Path(path).unlink()
            except OSError:
                pass
        except Exception:
            return f"![image]({url})" if url else None
        public = await freeimage.upload_png(http, data, f"gemini_{idx}.png")
        return f"![Generated Image {idx}]({public or url})"

    for idx, line in enumerate(await asyncio.gather(*(one(i, im) for i, im in enumerate(imgs)))):
        if line:
            md.append(line)
    return ("\n\n" + "\n\n".join(md)) if md else ""


def _full_text(out: Any) -> str:
    return getattr(out, "text", "") or ""


def _thoughts(out: Any) -> str:
    return getattr(out, "thoughts", None) or ""


def _usage(prompt: str, completion: str) -> dict:
    pt, ct = max(1, len(prompt) // 4), max(1, len(completion) // 4)
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


# ------------------------------------------------- conversation chaining ---


def _fingerprint(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            parts.append(f"{m.get('role', '')}:{c}")
        else:
            inner = "|".join(
                (p.get("text", "") if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text") else part_identity(p))
                for p in (c or [])
            )
            parts.append(f"{m.get('role', '')}:{inner}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:24]


async def _chat_turns(
    http: httpx.AsyncClient,
    sessions: SessionStore,
    messages: list[dict],
    conversation_id: str | None,
) -> tuple[str, list[tuple[str, list]], list]:
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
                    raise ValueError("last message has role assistant/tool; nothing new to send")
                prefix = "[System] " if role == "system" else ""
                turns.append((f"{prefix}{text}", uploads))
            return key, turns, state.metadata
        turns = []
        for m in messages:
            role, text, uploads = await message_to_turn(http, m)
            if role in ("assistant", "tool"):
                continue
            prefix = "[System] " if role == "system" else ""
            turns.append((f"{prefix}{text}", uploads))
        return key, turns or [(" ", [])], []

    if len(messages) > 1:
        prev_fp = _fingerprint(messages[:-1])
        prev = await sessions.get(f"fp:{prev_fp}")
        if prev is not None and prev.metadata:
            key = f"fp:{_fingerprint(messages)}"
            state, _ = await sessions.get_or_new(key)
            state.account = prev.account
            state.metadata = prev.metadata
            role, text, uploads = await message_to_turn(http, messages[-1])
            prefix = "[System] " if role == "system" else ""
            return key, [(f"{prefix}{text}", uploads)], prev.metadata

    key = f"fp:{_fingerprint(messages)}"
    await sessions.get_or_new(key)
    turns = []
    for m in messages:
        role, text, uploads = await message_to_turn(http, m)
        if role in ("assistant", "tool"):
            continue
        prefix = "[System] " if role == "system" else ""
        turns.append((f"{prefix}{text}", uploads))
    return key, turns or [(" ", [])], []


async def _pick_account(
    pool: AccountPool, sessions: SessionStore, key: str
) -> tuple[int, list]:
    """Sticky account per conversation state; fresh round-robin otherwise."""
    state = await sessions.get(key)
    if state is not None and state.account is not None:
        return state.account, state.metadata
    idx, _ = pool.pick()
    st, _ = await sessions.get_or_new(key)
    st.account = idx
    return idx, st.metadata


async def _save_state(
    sessions: SessionStore, key: str, account: int, metadata: list, full_fp: str | None = None
) -> None:
    state, _ = await sessions.get_or_new(key)
    state.account = account
    state.metadata = metadata
    if full_fp:
        await sessions.link(f"fp:{full_fp}", key)


# ------------------------------------------------------- chat completions ---

async def _chat_payload(
    req: Request, body: dict
) -> tuple[str | None, bool, list[dict], str | None]:
    model = body.get("model")
    stream = bool(body.get("stream", False))
    messages = body.get("messages") or []
    conversation_id = body.get("conversation_id")
    return model, stream, messages, conversation_id


async def _handle_chat(req: Request, body: dict) -> Any:
    pool, sessions, http = _pool(req), _sessions(req), _http(req)
    model_name, stream, messages, conversation_id = await _chat_payload(req, body)
    if not messages:
        return _err(400, "messages is required", "invalid_request_error")
    model, thinking = _resolve_model(pool, model_name)

    # Resolve key first (cheap), then serialize everything account-touching.
    key = f"conv:{conversation_id}" if conversation_id else None
    if key is None:
        try:
            key, turns, resume_meta = await _chat_turns(http, sessions, messages, conversation_id)
        except ValueError as exc:
            return _err(400, str(exc), "invalid_request_error")
        if not turns:
            return _err(400, "no turns to send", "invalid_request_error")
    else:
        # Stateless validation first: trailing assistant/tool carries nothing new,
        # regardless of whether server-side history exists yet.
        last_role = (messages[-1].get("role", "user") if isinstance(messages[-1], dict) else "user")
        if last_role in ("assistant", "tool"):
            return _err(400, "last message has role assistant/tool; nothing new to send", "invalid_request_error")
    convo_lock = await sessions.lock_for(key)
    async with convo_lock:
        if conversation_id:
            try:
                key, turns, resume_meta = await _chat_turns(http, sessions, messages, conversation_id)
            except ValueError as exc:
                return _err(400, str(exc), "invalid_request_error")
            if not turns:
                return _err(400, "no turns to send", "invalid_request_error")
        return await _handle_chat_locked(req, pool, sessions, http, key, turns, resume_meta, model_name, model, thinking, stream, messages)


async def _handle_chat_locked(req: Request, pool: AccountPool, sessions: SessionStore, http: httpx.AsyncClient, key: str, turns: list, resume_meta: list, model_name: str | None, model: str | None, thinking: bool, stream: bool, messages: list) -> Any:
    account, metadata = await _pick_account(pool, sessions, key)
    if resume_meta:
        metadata = resume_meta
    prompt_for_usage = "\n".join(t for t, _ in turns)

    if not stream:
        try:
            out, new_meta = await _run_turns(pool, account, model, thinking, turns, metadata)
        except Exception as exc:
            code, msg = _map_exception(exc)
            return _err(code, msg)
        md = await _gemini_images_to_markdown(http, out)
        text = _full_text(out) + md
        thoughts = _thoughts(out)
        await _save_state(sessions, key, account, new_meta, _fingerprint(messages))
        msg: dict[str, Any] = {"role": "assistant", "content": text}
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
                }
            ],
            "usage": _usage(prompt_for_usage, text),
            "conversation_id": key[5:] if key.startswith("conv:") else None,
        }

    async def gen():
        cid = _new_id("chatcmpl-")
        sent_role = False
        full = ""
        try:
            async for kind, payload in _run_turns_stream(
                pool, account, model, thinking, turns, metadata
            ):
                if kind == "delta":
                    if not sent_role:
                        sent_role = True
                        yield _sse(
                            {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": _now(),
                                "model": model_name or "gemini-flash",
                                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                            }
                        )
                    full += payload
                    yield _sse(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": _now(),
                            "model": model_name or "gemini-flash",
                            "choices": [{"index": 0, "delta": {"content": payload}, "finish_reason": None}],
                        }
                    )
                else:
                    out, new_meta = payload
                    md = await _gemini_images_to_markdown(http, out)
                    if md:
                        full += md
                        yield _sse(
                            {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": _now(),
                                "model": model_name or "gemini-flash",
                                "choices": [{"index": 0, "delta": {"content": md}, "finish_reason": None}],
                            }
                        )
                    await _save_state(sessions, key, account, new_meta, _fingerprint(messages))
                    yield _sse(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": _now(),
                            "model": model_name or "gemini-flash",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": _usage(prompt_for_usage, full),
                        }
                    )
        except Exception as exc:
            yield _sse({"error": {"message": str(exc) or type(exc).__name__}})
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: Request):
    try:
        body = await req.json()
    except Exception:
        return _err(400, "invalid JSON body", "invalid_request_error")
    return await _handle_chat(req, body)


# ------------------------------------------------------------- responses ---

def _norm_input_item(item: Any) -> str:
    if not isinstance(item, dict):
        return json.dumps(item, sort_keys=True, default=str)
    itype = item.get("type", "message")
    if itype != "message":
        return json.dumps({k: item.get(k) for k in ("type", "id")}, sort_keys=True, default=str)
    parts: list[str] = []
    for p in item.get("content", []) or []:
        if not isinstance(p, dict):
            continue
        if p.get("type") in ("input_text", "output_text", "text"):
            parts.append(p.get("text", ""))
        else:
            parts.append(part_identity(p))
    return f"{item.get('role', '')}:{('|'.join(parts))}"


async def _responses_turns(
    http: httpx.AsyncClient,
    sessions: SessionStore,
    items: list[dict],
    instructions: str | None,
    prev_response_id: str | None,
    conversation: str | None,
) -> tuple[str, list[tuple[str, list]], list, str | None]:
    """Resolve (state_key, NEW turns only, resume metadata, prev text) for Responses.

    Resumed server-side history is never replayed: only trailing items not
    already covered by the stored input norm are sent.
    """
    if conversation:
        key = f"conv:{conversation}"
        state, created = await sessions.get_or_new(key)
        if not created and state.metadata:
            prev_norm: list[str] = (await sessions.get_norm(key)) or []
            new_norm = [_norm_input_item(i) for i in items]
            cut = 0
            for a, b in zip(prev_norm, new_norm):
                if a != b:
                    break
                cut += 1
            fresh = items[cut:]
            turns = []
            for it in fresh:
                role, text, uploads = await message_to_turn(
                    http, {"role": it.get("role", "user"), "content": it.get("content", "")}
                )
                if role in ("assistant", "tool"):
                    for up, _ in uploads:
                        up.cleanup()
                    continue
                turns.append((text, uploads))
            if instructions:
                turns.insert(0, (f"[System] {instructions}", []))
            return key, turns, state.metadata, None
        turns = []
        if instructions:
            turns.append((f"[System] {instructions}", []))
        for it in items:
            role, text, uploads = await message_to_turn(
                http, {"role": it.get("role", "user"), "content": it.get("content", "")}
            )
            if role == "assistant":
                continue
            turns.append((text, uploads))
        return key, turns or [(" ", [])], [], None

    if prev_response_id:
        saved = await sessions.get_response(prev_response_id)
        if saved and saved.get("_state"):
            key = saved["_state"]
            state = await sessions.get(key)
            old_norm = saved.get("_norm") or []
            new_norm = [_norm_input_item(i) for i in items]
            cut = 0
            for a, b in zip(old_norm, new_norm):
                if a != b:
                    break
                cut += 1
            fresh = items[cut:]
            turns = []
            for it in fresh:
                role, text, uploads = await message_to_turn(
                    http, {"role": it.get("role", "user"), "content": it.get("content", "")}
                )
                if role in ("assistant", "tool"):
                    for up, _ in uploads:
                        up.cleanup()
                    continue
                turns.append((text, uploads))
            if instructions:
                turns.insert(0, (f"[System] {instructions}", []))
            return key, turns, (state.metadata if state else []), saved.get("_text")

    key = f"resp:{uuid.uuid4().hex[:16]}:{hashlib.sha256('|'.join(_norm_input_item(i) for i in items).encode()).hexdigest()[:12]}"
    await sessions.get_or_new(key)
    turns = []
    if instructions:
        turns.append((f"[System] {instructions}", []))
    for it in items:
        role, text, uploads = await message_to_turn(
            http, {"role": it.get("role", "user"), "content": it.get("content", "")}
        )
        if role == "assistant":
            continue
        turns.append((text, uploads))
    return key, turns or [(" ", [])], [], None


def _response_object(
    rid: str,
    model_name: str,
    text: str,
    conv_id: str,
    prev_id: str | None,
    prompt: str,
    status: str = "completed",
) -> dict:
    return {
        "id": rid,
        "object": "response",
        "created_at": _now(),
        "model": model_name or "gemini-flash",
        "status": status,
        "output": [
            {
                "type": "message",
                "id": _new_id("msg_"),
                "status": status,
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": max(1, len(prompt) // 4),
            "output_tokens": max(1, len(text) // 4),
            "total_tokens": max(2, (len(prompt) + len(text)) // 4),
        },
        "conversation": {"id": conv_id},
        "previous_response_id": prev_id,
    }


async def _handle_responses(req: Request, body: dict) -> Any:
    pool, sessions, http = _pool(req), _sessions(req), _http(req)
    model_name = body.get("model")
    raw_input = body.get("input", "")
    instructions = body.get("instructions")
    prev_id = body.get("previous_response_id")
    conv_param = body.get("conversation")
    conv_id = conv_param.get("id") if isinstance(conv_param, dict) else conv_param
    stream = bool(body.get("stream", False))
    model, thinking = _resolve_model(pool, model_name)

    if isinstance(raw_input, str):
        prompt: str = raw_input
        items: list[dict] = []
        if prev_id or conv_id:
            as_items = (
                [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}]
                if prompt.strip()
                else []
            )
            items = as_items
            pre_turns: list | None = None
            key = None
        else:
            pre_key = f"resp:{uuid.uuid4().hex[:16]}:{hashlib.sha256(prompt.encode()).hexdigest()[:12]}"
            pre_turns = ([(f"[System] {instructions}\n\n{prompt}", [])] if instructions else [(prompt, [])])
            key = pre_key
            await sessions.get_or_new(key)
            turns, resume_meta = pre_turns, []
    else:
        items = list(raw_input or [])
        pre_turns = None
        key = None

    if key is None:
        # Stateless validation: empty item list never reaches Gemini.
        if not items and pre_turns is None:
            return _err(400, "no input items to send", "invalid_request_error")
        # Peek the key without downloading files: derive from stored norm state.
        if conv_id:
            key = f"conv:{conv_id}"
        elif prev_id:
            saved = await sessions.get_response(prev_id)
            key = saved.get("_state") if saved and saved.get("_state") else f"pending:{prev_id}"
        else:
            key = f"resp:{uuid.uuid4().hex[:16]}:{hashlib.sha256('|'.join(_norm_input_item(i) for i in items).encode()).hexdigest()[:12]}"
    convo_lock = await sessions.lock_for(key)
    async with convo_lock:
        if pre_turns is None:
            try:
                key, turns, resume_meta, _ = await _responses_turns(
                    http, sessions, items, instructions, prev_id, conv_id
                )
            except ValueError as exc:
                return _err(400, str(exc), "invalid_request_error")
        if not turns:
            return _err(400, "no input items to send", "invalid_request_error")
        return await _handle_responses_locked(req, pool, sessions, http, key, turns, resume_meta, items, model_name, model, thinking, stream, instructions, prev_id)


async def _handle_responses_locked(req: Request, pool: AccountPool, sessions: SessionStore, http: httpx.AsyncClient, key: str, turns: list, resume_meta: list, items: list, model_name: str | None, model: str | None, thinking: bool, stream: bool, instructions: str | None, prev_id: str | None) -> Any:
    account, metadata = await _pick_account(pool, sessions, key)
    if resume_meta:
        metadata = resume_meta
    prompt_for_usage = "\n".join(t for t, _ in turns)
    conv_out = key[5:] if key.startswith("conv:") else key

    if not stream:
        try:
            out, new_meta = await _run_turns(pool, account, model, thinking, turns, metadata)
        except Exception as exc:
            code, msg = _map_exception(exc)
            return _err(code, msg)
        md = await _gemini_images_to_markdown(http, out)
        text = _full_text(out) + md
        await _save_state(sessions, key, account, new_meta)
        await sessions.set_norm(key, [_norm_input_item(i) for i in items])
        obj = _response_object(rid, model_name, text, conv_out, prev_id, prompt_for_usage)
        obj["_state"] = key
        obj["_text"] = text
        obj["_norm"] = [_norm_input_item(i) for i in items]
        await sessions.save_response(rid, obj)
        return {k: v for k, v in obj.items() if not k.startswith("_")}

    async def gen():
        rid = _new_id("resp_")
        yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': {'id': rid, 'object': 'response', 'status': 'in_progress', 'model': model_name or 'gemini-flash'}})}\n\n"
        full = ""
        try:
            async for kind, payload in _run_turns_stream(
                pool, account, model, thinking, turns, metadata
            ):
                if kind == "delta":
                    full += payload
                    yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'item_id': rid, 'delta': payload})}\n\n"
                else:
                    out, new_meta = payload
                    md = await _gemini_images_to_markdown(http, out)
                    if md:
                        full += md
                        yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'item_id': rid, 'delta': md})}\n\n"
                    await _save_state(sessions, key, account, new_meta)
                    await sessions.set_norm(key, [_norm_input_item(i) for i in items])
                    obj = _response_object(rid, model_name, full, conv_out, prev_id, prompt_for_usage)
                    obj["_state"] = key
                    obj["_text"] = full
                    obj["_norm"] = [_norm_input_item(i) for i in items]
                    await sessions.save_response(rid, obj)
                    public = {k: v for k, v in obj.items() if not k.startswith("_")}
                    yield f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': public})}\n\n"
        except Exception as exc:
            yield f"event: response.failed\ndata: {json.dumps({'type': 'response.failed', 'error': str(exc)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/responses")
@app.post("/responses")
async def create_response(req: Request):
    try:
        body = await req.json()
    except Exception:
        return _err(400, "invalid JSON body", "invalid_request_error")
    return await _handle_responses(req, body)


@app.get("/v1/responses/{rid}")
@app.get("/responses/{rid}")
async def get_response(req: Request, rid: str):
    saved = await _sessions(req).get_response(rid)
    if not saved:
        return _err(404, f"response {rid} not found", "invalid_request_error")
    return {k: v for k, v in saved.items() if not k.startswith("_")}


# ----------------------------------------------------------------- models ---

@app.get("/v1/models")
@app.get("/models")
async def list_models(req: Request):
    return {"object": "list", "data": _model_list(_pool(req))}


# ------------------------------------------------------------------ files ---

@app.post("/v1/files")
async def upload_file(req: Request):
    form = await req.form()
    file = form.get("file")
    purpose = str(form.get("purpose", "assistants"))
    if file is None or not hasattr(file, "read"):
        return _err(400, "file is required (multipart)", "invalid_request_error")
    data = await file.read()
    filename = getattr(file, "filename", None) or "upload.bin"
    import mimetypes as _m

    mime = _m.guess_type(filename)[0] or "application/octet-stream"
    try:
        fid = UploadStore.put(data, mime, filename)
    except ValueError as exc:
        return _err(413, str(exc), "invalid_request_error")
    return {
        "id": fid,
        "object": "file",
        "bytes": len(data),
        "created_at": _now(),
        "filename": filename,
        "purpose": purpose,
    }


@app.get("/v1/files")
async def list_files():
    return {
        "object": "list",
        "data": [
            {"id": fid, "object": "file", "bytes": len(b), "created_at": 0, "filename": fn, "purpose": "assistants"}
            for fid, (b, _, fn) in UploadStore._files.items()
        ],
    }


@app.get("/v1/files/{fid}")
async def retrieve_file(fid: str):
    stored = UploadStore.get(fid)
    if not stored:
        return _err(404, f"file {fid} not found", "invalid_request_error")
    data, _, filename = stored
    return {"id": fid, "object": "file", "bytes": len(data), "created_at": 0, "filename": filename, "purpose": "assistants"}


@app.delete("/v1/files/{fid}")
async def delete_file(fid: str):
    if not UploadStore.delete(fid):
        return _err(404, f"file {fid} not found", "invalid_request_error")
    return {"id": fid, "object": "file", "deleted": True}


@app.get("/v1/files/{fid}/content")
async def file_content(fid: str):
    from fastapi.responses import Response as RawResponse

    stored = UploadStore.get(fid)
    if not stored:
        return _err(404, f"file {fid} not found", "invalid_request_error")
    data, mime, _ = stored
    return RawResponse(content=data, media_type=mime)


# ----------------------------------------------------------------- images ---

async def _handle_image_gen(req: Request, body: dict) -> Any:
    pool, sessions, http = _pool(req), _sessions(req), _http(req)
    prompt = body.get("prompt", "")
    if not prompt.strip():
        return _err(400, "prompt is required", "invalid_request_error")
    model_name = body.get("model")
    model, thinking = _resolve_model(pool, model_name)
    key = f"img:{uuid.uuid4().hex[:16]}"
    account, _ = await _pick_account(pool, sessions, key)
    try:
        out, _ = await _run_turns(
            pool, account, model, thinking, [(prompt, [])], []
        )
    except Exception as exc:
        code, msg = _map_exception(exc)
        return _err(code, msg)
    imgs = list(getattr(out, "images", None) or [])
    if not imgs:
        return _err(500, "Gemini returned no image for this prompt")
    data: list[dict] = []
    for idx, img in enumerate(imgs):
        url = (getattr(img, "url", "") or "").strip()
        try:
            path = await img.save(path=str(IMG_TMP), filename=f"gen_{uuid.uuid4().hex[:8]}")
            raw = Path(path).read_bytes()
            try:
                Path(path).unlink()
            except OSError:
                pass
            public = (await freeimage.upload_png(http, raw, f"gemini_{idx}.png") or "").strip()
            final = public or url
            if final:
                data.append({"url": final, "revised_prompt": prompt})
        except Exception:
            if url:
                data.append({"url": url, "revised_prompt": prompt})
    if not data:
        return _err(500, "failed to retrieve generated image bytes")
    return {"created": _now(), "data": data}


@app.post("/v1/images/generations")
@app.post("/images/generations")
async def image_generations(req: Request):
    try:
        body = await req.json()
    except Exception:
        return _err(400, "invalid JSON body", "invalid_request_error")
    return await _handle_image_gen(req, body)


# ------------------------------------------------------------------ misc ---

@app.get("/health")
@app.get("/")
async def health(req: Request):
    try:
        n = len(_pool(req)._entries)
    except Exception:
        n = 0
    return {"status": "ok", "accounts": n, "freeimage": bool(freeimage_api_key())}


@app.get("/v1/conversations/{cid}")
async def get_conversation(req: Request, cid: str):
    state = await _sessions(req).get(f"conv:{cid}")
    if state is None:
        return _err(404, f"conversation {cid} not found", "invalid_request_error")
    return {"id": cid, "object": "conversation", "has_history": bool(state.metadata)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
