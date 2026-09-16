# Copyright 2026 gem2oai contributors.
"""OpenAI message content -> (prompt text, Gemini file uploads).

Accepts every input shape both APIs allow: plain strings, content part lists
(text/image/file/image_url/input_file/input_image), attachments, and direct
base64 data URLs. Downloads http(s) URLs; filenames preserve extensions so
Gemini detects images, JSON, txt, py, etc.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import httpx

from store import connect

if TYPE_CHECKING:
    import sqlite3

_DATA_URL_PREFIX = "data:"
_STALE_MAX_AGE = 86400
_TEXT_PREVIEW_LEN = 160
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "gem2oai-uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# Reap files from crashed runs / leaked turns (older than 1 day).
for _stale in _UPLOAD_DIR.glob("*"):
    with suppress(OSError):
        if _stale.is_file() and _stale.stat().st_mtime < (time.time() - _STALE_MAX_AGE):
            _stale.unlink()
_EXT_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/heic": ".heic",
    "application/json": ".json",
    "text/plain": ".txt",
    "text/x-python": ".py",
    "text/csv": ".csv",
    "text/markdown": ".md",
    "application/pdf": ".pdf",
}


def _ext_for(mime: str | None, name: str) -> str:
    """Pick a file extension for a staged upload from its name or MIME type."""
    if "." in name.rsplit("/", 1)[-1]:
        return ""
    if mime and mime in _EXT_BY_MIME:
        return _EXT_BY_MIME[mime]
    return mimetypes.guess_extension((mime or "").split(";")[0].strip()) or ".bin"


async def _download(client: httpx.AsyncClient, url: str) -> tuple[bytes, str | None]:
    """Download an attachment URL with a browser-like Accept header."""
    r = await client.get(
        url,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    if r.status_code in (401, 403):
        msg = f"URL download blocked ({r.status_code}): {url}"
        raise RuntimeError(msg)
    r.raise_for_status()
    content_type = r.headers.get("content-type")
    return r.content, content_type if isinstance(content_type, str) else None


class UploadFile:
    """A staged temp file for Gemini upload.

    gemini-webapi derives the upload filename AND content-type from the path
    it is given (`parse_file_name` + `mimetypes.guess_type`), and ignores
    `BytesIO.name`. So uploads MUST be real paths with correct extensions.
    Call `cleanup()` after the turn is sent.
    """

    __slots__ = ("mime", "path")

    def __init__(self, path: Path, mime: str) -> None:
        """Record the staged path and resolved MIME type."""
        self.path = path
        self.mime = mime

    def __str__(self) -> str:
        """Return the staged path as a string."""
        return str(self.path)

    def cleanup(self) -> None:
        """Delete the staged temp file, ignoring missing-file races."""
        with suppress(OSError):
            self.path.unlink(missing_ok=True)


def _clean_filename(filename: str | None) -> str:
    """Strip URL query parts from an upload filename, defaulting to upload."""
    return (filename or "upload").rsplit("/", 1)[-1].rsplit("?", 1)[
        0
    ].strip() or "upload"


def _as_upload(
    data: bytes,
    mime: str | None,
    filename: str | None,
) -> tuple[UploadFile, str]:
    """Stage bytes to a real temp path so Gemini detects name and MIME type."""
    base = _clean_filename(filename)
    if "." not in base.rsplit("/", 1)[-1]:
        base += _ext_for(mime, "")
    path = _UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{base}"
    path.write_bytes(data)
    resolved = mime or mimetypes.guess_type(base)[0] or "application/octet-stream"
    return UploadFile(path, resolved), resolved


def _part_filename(part: dict[str, Any], fallback: dict[str, Any]) -> str | None:
    """Pick the first string filename from a part or its nested file dict."""
    name = part.get("filename")
    if isinstance(name, str) and name:
        return name
    nested = fallback.get("filename")
    return nested if isinstance(nested, str) and nested else None


def _stored_upload(file_id: object) -> tuple[UploadFile, str]:
    """Re-stage a Files API upload by id."""
    stored = UploadStore.get(file_id) if isinstance(file_id, str) else None
    if stored is None:
        msg = f"unknown file_id: {file_id}"
        raise ValueError(msg)
    return _as_upload(stored[0], stored[1], stored[2])


async def _file_part_to_upload(
    http: httpx.AsyncClient,
    part: dict[str, Any],
    nested: dict[str, Any],
) -> tuple[UploadFile, str] | None:
    """Convert an input_file/file part to a staged upload."""
    file_id, file_data = nested.get("file_id"), nested.get("file_data")
    if file_id:
        return _stored_upload(file_id)
    if isinstance(file_data, str) and file_data:
        data, mime = _decode_data_url(file_data)
        return _as_upload(data, mime, _part_filename(part, nested))
    url = nested.get("url")
    if isinstance(url, str) and url:
        data, mime = await _download(http, url)
        return _as_upload(
            data,
            mime,
            _part_filename(part, nested) or url,
        )
    return None


def _image_url_from_part(part: dict[str, Any]) -> str | None:
    """Extract the image URL from the shapes image parts allow."""
    img = part.get("image_url", part.get("image", part))
    url: str | None = None
    if isinstance(img, dict):
        u = img.get("url")
        url = u if isinstance(u, str) else None
    elif isinstance(img, str):
        url = img
    if not url and isinstance(part.get("image_url"), str):
        url = part["image_url"]
    if not url:
        direct = part.get("url")
        url = direct if isinstance(direct, str) else None
    return url


async def _image_part_to_upload(
    http: httpx.AsyncClient,
    part: dict[str, Any],
) -> tuple[UploadFile, str] | None:
    """Convert an image part (URL or data URL) to a staged upload."""
    url = _image_url_from_part(part)
    if not url:
        return None
    if url.startswith(_DATA_URL_PREFIX):
        data, mime = _decode_data_url(url)
        filename = part.get("filename")
        return _as_upload(
            data,
            mime,
            filename if isinstance(filename, str) else None,
        )
    data, mime = await _download(http, url)
    tail = url.split("?", 1)[0].rsplit("/", 1)[-1].split("#", 1)[0]
    filename = part.get("filename")
    return _as_upload(
        data,
        mime,
        (filename if isinstance(filename, str) else None) or tail,
    )


async def _part_to_file(
    http: httpx.AsyncClient,
    part: dict[str, Any],
) -> tuple[UploadFile, str] | None:
    """Convert one content part to a staged upload, or None for text parts."""
    ptype = part.get("type", "")
    if ptype in ("input_file", "file"):
        nested = part.get("file", part)
        if not isinstance(nested, dict):
            return None
        return await _file_part_to_upload(http, part, nested)
    if ptype in ("input_image", "image_url", "image"):
        return await _image_part_to_upload(http, part)
    return None


def _decode_data_url(url: str) -> tuple[bytes, str | None]:
    """Split a base64 data URL into (bytes, MIME type)."""
    header, sep, payload = url.partition(",")
    if not sep or not payload:
        msg = "invalid data URL (missing ',' payload)"
        raise ValueError(msg)
    mime = header.split(";")[0].split(":")[1] if ":" in header else None
    data = base64.b64decode(payload, validate=False)
    return data, mime


class UploadedFile(NamedTuple):
    """Stored Files API entry: bytes plus metadata."""

    data: bytes
    mime: str
    filename: str
    purpose: str
    created: int


class UploadStore:
    """Files API storage: id -> UploadedFile. Bounded (FIFO cap).

    Write-through to SQLite when bound, so uploaded files survive a restart.
    """

    _files: ClassVar[dict[str, UploadedFile]] = {}
    _db: ClassVar[sqlite3.Connection | None] = None
    _cap: ClassVar[int] = 200
    MAX_BYTES: ClassVar[int] = 25 * 1024 * 1024

    @classmethod
    def bind(cls, path: Path | str) -> None:
        """Persist files to the SQLite database at path."""
        cls.close()
        cls._db = connect(path)

    @classmethod
    def close(cls) -> None:
        """Flush and detach the SQLite handle, keeping the memory cache."""
        if cls._db is not None:
            try:
                cls._db.commit()
            finally:
                cls._db.close()
            cls._db = None

    @classmethod
    def _evict_db(cls) -> list[str]:
        """Drop oldest file rows beyond the cap; return evicted ids."""
        if cls._db is None:
            return []
        total_row = cls._db.execute("SELECT COUNT(*) FROM files").fetchone()
        total = int(total_row[0]) if total_row else 0
        if total <= cls._cap:
            return []
        rows = cls._db.execute(
            "SELECT id FROM files ORDER BY ROWID ASC LIMIT ?",
            (total - cls._cap,),
        ).fetchall()
        evicted = [str(r[0]) for r in rows]
        cls._db.execute(
            "DELETE FROM files WHERE id IN (SELECT id FROM files"
            " ORDER BY ROWID ASC LIMIT ?)",
            (len(evicted),),
        )
        return evicted

    @classmethod
    def put(
        cls,
        data: bytes,
        mime: str,
        filename: str,
        purpose: str = "assistants",
    ) -> str:
        """Store file bytes, returning a file id (persisted when bound)."""
        if len(data) > cls.MAX_BYTES:
            msg = f"file too large ({len(data)} bytes > {cls.MAX_BYTES})"
            raise ValueError(msg)
        while len(cls._files) >= cls._cap:
            cls._files.pop(next(iter(cls._files)))
        fid = f"file_{uuid.uuid4().hex[:24]}"
        entry = UploadedFile(data, mime, filename, purpose, int(time.time()))
        cls._files[fid] = entry
        if cls._db is not None:
            cls._db.execute(
                "INSERT INTO files(id, data, mime, filename, purpose, created_at)"
                " VALUES(?, ?, ?, ?, ?, ?)",
                (fid, data, mime, filename, purpose, entry.created),
            )
            for evicted in cls._evict_db():
                cls._files.pop(evicted, None)
            cls._db.commit()
        return fid

    @classmethod
    def get(cls, fid: str) -> UploadedFile | None:
        """Fetch file bytes, falling back to SQLite after a restart."""
        hit = cls._files.get(fid)
        if hit is not None:
            return hit
        if cls._db is None:
            return None
        row = cls._db.execute(
            "SELECT data, mime, filename, purpose, created_at FROM files WHERE id=?",
            (fid,),
        ).fetchone()
        if row is None:
            return None
        entry = UploadedFile(
            bytes(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            int(row[4]),
        )
        while len(cls._files) >= cls._cap:
            cls._files.pop(next(iter(cls._files)))
        cls._files[fid] = entry
        return entry

    @classmethod
    def delete(cls, fid: str) -> bool:
        """Delete a file, returning False when the id is unknown."""
        found = cls._files.pop(fid, None) is not None
        if cls._db is not None:
            cur = cls._db.execute("DELETE FROM files WHERE id=?", (fid,))
            cls._db.commit()
            return found or cur.rowcount > 0
        return found

    @classmethod
    def list_all(cls) -> list[tuple[str, int, str, str, int]]:
        """List (id, size, filename, purpose, created_at), oldest first."""
        if cls._db is not None:
            return [
                (str(r[0]), int(r[1]), str(r[2]), str(r[3]), int(r[4]))
                for r in cls._db.execute(
                    "SELECT id, LENGTH(data), filename, purpose, created_at FROM files"
                    " ORDER BY created_at ASC, id ASC",
                ).fetchall()
            ]
        return [
            (fid, len(entry.data), entry.filename, entry.purpose, entry.created)
            for fid, entry in cls._files.items()
        ]


TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})


def _append_text_part(texts: list[str], part: dict[str, Any]) -> None:
    """Append non-empty text from a content part."""
    text = part.get("text", "")
    if isinstance(text, str) and text.strip():
        texts.append(text)


async def _collect_part_upload(
    http: httpx.AsyncClient,
    part: dict[str, Any],
    texts: list[str],
    uploads: list[tuple[UploadFile, str]],
) -> None:
    """Resolve one non-text part to an upload or an inline skip note."""
    try:
        up = await _part_to_file(http, part)
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        texts.append(f"[Attachment skipped: {exc}]")
        return
    if up:
        uploads.append(up)


async def message_to_turn(
    http: httpx.AsyncClient,
    msg: dict[str, Any],
) -> tuple[str, str, list[tuple[UploadFile, str]]]:
    """Convert one OpenAI message to (role, text, uploads)."""
    role = msg.get("role", "user")
    role_str = role if isinstance(role, str) else "user"
    content = msg.get("content", "")
    texts: list[str] = []
    uploads: list[tuple[UploadFile, str]] = []
    if isinstance(content, str):
        if content.strip():
            texts.append(content)
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in TEXT_PART_TYPES:
                _append_text_part(texts, part)
            elif part.get("type") == "refusal":
                refusal = part.get("refusal", "")
                texts.append(
                    f"[refusal] {refusal}" if isinstance(refusal, str) else "[refusal]",
                )
            else:
                await _collect_part_upload(http, part, texts, uploads)
    for att in msg.get("attachments") or []:
        if isinstance(att, dict):
            await _collect_part_upload(
                http,
                {"type": "input_file", **att},
                texts,
                uploads,
            )
    return role_str, "\n\n".join(texts), uploads


def _short(val: str) -> str:
    """Shorten long identity values to a stable sha256 digest."""
    if len(val) <= _TEXT_PREVIEW_LEN:
        return val
    return hashlib.sha256(val.encode()).hexdigest()[: _TEXT_PREVIEW_LEN // 10]


def _identity_from_keys(part: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Return the first key=value identity found in a part dict."""
    for key in keys:
        val = part.get(key)
        if isinstance(val, str) and val:
            return f"{key}={_short(val)}"
    return None


def part_identity(part: object) -> str:
    """Return cheap file identity (no downloads) for conversation fingerprinting."""
    if not isinstance(part, dict):
        return ""
    direct = _identity_from_keys(part, ("filename", "file_id", "file_data", "url"))
    if direct:
        return direct
    for key in ("file", "image_url", "image"):
        nested = part.get(key)
        if isinstance(nested, dict):
            inner = _identity_from_keys(
                nested,
                ("filename", "file_id", "file_data", "url"),
            )
            if inner:
                return f"{key}.{inner}"
        elif isinstance(nested, str) and nested:
            return f"{key}={_short(nested)}"
    ptype = part.get("type", "")
    return ptype if isinstance(ptype, str) else ""


def _role_prefix(role: object) -> str:
    """Format a message role as a prompt prefix, empty for plain users."""
    if not isinstance(role, str) or role in ("user", ""):
        return ""
    return f"[{role.capitalize()}] "


def _append_prompt_text(texts: list[str], prefix: str, text: object) -> None:
    """Append non-empty text with its role prefix."""
    if isinstance(text, str) and text.strip():
        texts.append(f"{prefix}{text}" if prefix else text)


async def _collect_prompt_part(
    http: httpx.AsyncClient,
    part: dict[str, Any],
    prefix: str,
    texts: list[str],
    uploads: list[tuple[UploadFile, str]],
) -> None:
    """Collect one prompt part into text or a staged upload."""
    ptype = part.get("type", "")
    if ptype in ("text", "input_text", "output_text"):
        _append_prompt_text(texts, prefix, part.get("text", ""))
    elif ptype == "refusal":
        _append_prompt_text(texts, prefix, f"[refusal] {part.get('refusal', '')}")
    else:
        await _collect_part_upload(http, part, texts, uploads)


async def _collect_prompt_attachments(
    http: httpx.AsyncClient,
    msg: dict[str, Any],
    uploads: list[tuple[UploadFile, str]],
) -> None:
    """Stage every attachment dict on a message."""
    for att in msg.get("attachments") or []:
        if isinstance(att, dict):
            up = await _part_to_file(http, {"type": "input_file", **att})
            if up:
                uploads.append(up)


async def openai_messages_to_prompt(
    http: httpx.AsyncClient,
    messages: list[dict[str, Any]],
) -> tuple[str, list[tuple[UploadFile, str]]]:
    """Flatten one OpenAI messages array into a single prompt turn + file uploads.

    Multi-turn across requests is NOT rebuilt here: the Gemini ChatSession already
    holds history server-side. System/developer messages become a [System] preamble
    on the first turn of a conversation.
    """
    texts: list[str] = []
    uploads: list[tuple[UploadFile, str]] = []

    for msg in messages:
        prefix = _role_prefix(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, str):
            _append_prompt_text(texts, prefix, content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    await _collect_prompt_part(http, part, prefix, texts, uploads)
        await _collect_prompt_attachments(http, msg, uploads)

    prompt = "\n\n".join(texts) or " "
    return prompt, uploads


async def responses_input_to_prompt(
    http: httpx.AsyncClient,
    raw_input: object,
) -> tuple[str, list[tuple[UploadFile, str]]]:
    """Flatten Responses API input (string or item list) to a prompt + uploads."""
    if isinstance(raw_input, str):
        return raw_input, []
    if not isinstance(raw_input, list):
        return " ", []
    messages: list[dict[str, Any]] = []
    for item in raw_input:
        if not isinstance(item, dict):
            continue
        itype = item.get("type", "message")
        if itype == "message":
            messages.append(
                {"role": item.get("role", "user"), "content": item.get("content", "")},
            )
        elif itype == "image_generation_call":
            messages.append({"role": "user", "content": "Generate an image."})
    return await openai_messages_to_prompt(http, messages)
