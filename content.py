"""OpenAI message content -> (prompt text, Gemini file uploads).

Accepts every input shape both APIs allow: plain strings, content part lists
(text/image/file/image_url/input_file/input_image), attachments, and direct
base64 data URLs. Downloads http(s) URLs; filenames preserve extensions so
Gemini detects images, JSON, txt, py, etc.
"""

from __future__ import annotations

import base64
import mimetypes
import tempfile
import uuid
from pathlib import Path

import httpx

_DATA_URL_PREFIX = "data:"
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "gem2oai-uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
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
    if "." in name.rsplit("/", 1)[-1]:
        return ""
    if mime and mime in _EXT_BY_MIME:
        return _EXT_BY_MIME[mime]
    return mimetypes.guess_extension((mime or "").split(";")[0].strip()) or ".bin"


async def _download(client: httpx.AsyncClient, url: str) -> tuple[bytes, str | None]:
    r = await client.get(
        url,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    if r.status_code in (401, 403):
        raise RuntimeError(f"URL download blocked ({r.status_code}): {url}")
    r.raise_for_status()
    return r.content, r.headers.get("content-type")


class UploadFile:
    """A staged temp file for Gemini upload.

    gemini-webapi derives the upload filename AND content-type from the path
    it is given (`parse_file_name` + `mimetypes.guess_type`), and ignores
    `BytesIO.name`. So uploads MUST be real paths with correct extensions.
    Call `cleanup()` after the turn is sent.
    """

    __slots__ = ("mime", "path")

    def __init__(self, path: Path, mime: str):
        self.path = path
        self.mime = mime

    def __str__(self) -> str:
        return str(self.path)

    def cleanup(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def _as_upload(
    data: bytes, mime: str | None, filename: str | None
) -> tuple[UploadFile, str]:
    base = (filename or "upload").rsplit("/", 1)[-1].rsplit("?", 1)[0].strip() or "upload"
    if "." not in base.rsplit("/", 1)[-1]:
        base += _ext_for(mime, "")
    path = _UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{base}"
    path.write_bytes(data)
    resolved = mime or mimetypes.guess_type(base)[0] or "application/octet-stream"
    return UploadFile(path, resolved), resolved


async def _part_to_file(
    http: httpx.AsyncClient, part: dict
) -> tuple[UploadFile, str] | None:
    ptype = part.get("type", "")
    if ptype in ("input_file", "file"):
        f = part.get("file", part)
        file_id, file_data = f.get("file_id"), f.get("file_data")
        if file_id:
            stored = UploadStore.get(file_id)
            if stored:
                return _as_upload(stored[0], stored[1], stored[2])
            return None
        if file_data:
            data, mime = _decode_data_url(file_data)
            return _as_upload(data, mime, part.get("filename") or f.get("filename"))
        if f.get("url"):
            data, mime = await _download(http, f["url"])
            return _as_upload(
                data, mime, part.get("filename") or f.get("filename") or f["url"]
            )
        return None
    if ptype in ("input_image", "image_url", "image"):
        img = part.get("image_url", part.get("image", part))
        url = img.get("url") if isinstance(img, dict) else img if isinstance(part.get("image_url"), str) else None
        if not url:
            url = part.get("url") if isinstance(part.get("url"), str) else None
        if not url:
            return None
        if url.startswith(_DATA_URL_PREFIX):
            data, mime = _decode_data_url(url)
            return _as_upload(data, mime, part.get("filename"))
        data, mime = await _download(http, url)
        return _as_upload(data, mime, part.get("filename") or url.split("?")[0].rsplit("/", 1)[-1])
    return None


def _decode_data_url(url: str) -> tuple[bytes, str | None]:
    header, _, payload = url.partition(",")
    mime = header.split(";")[0].split(":")[1] if ":" in header else None
    data = base64.b64decode(payload, validate=False)
    return data, mime


class UploadStore:
    """Files API storage: id -> (bytes, mime, filename)."""

    _files: dict[str, tuple[bytes, str, str]] = {}

    @classmethod
    def put(cls, data: bytes, mime: str, filename: str) -> str:
        fid = f"file_{uuid.uuid4().hex[:24]}"
        cls._files[fid] = (data, mime, filename)
        return fid

    @classmethod
    def get(cls, fid: str) -> tuple[bytes, str, str] | None:
        return cls._files.get(fid)

    @classmethod
    def delete(cls, fid: str) -> bool:
        return cls._files.pop(fid, None) is not None


TEXT_PART_TYPES = {"text", "input_text", "output_text"}


async def message_to_turn(
    http: httpx.AsyncClient, msg: dict
) -> tuple[str, str, list[tuple[UploadFile, str]]]:
    """Single OpenAI message -> (role, text, uploads). No history flattening."""
    role = msg.get("role", "user")
    content = msg.get("content", "")
    texts: list[str] = []
    uploads: list[tuple[UploadFile, str]] = []
    if isinstance(content, str):
        if content.strip():
            texts.append(content)
    else:
        for part in content or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") in TEXT_PART_TYPES:
                t = part.get("text", "")
                if isinstance(t, str) and t.strip():
                    texts.append(t)
            elif part.get("type") == "refusal":
                texts.append(f"[refusal] {part.get('refusal', '')}")
            else:
                try:
                    up = await _part_to_file(http, part)
                except Exception as exc:
                    texts.append(f"[Attachment skipped: {exc}]")
                    continue
                if up:
                    uploads.append(up)
    for att in msg.get("attachments") or []:
        if isinstance(att, dict):
            try:
                up = await _part_to_file(http, {"type": "input_file", **att})
            except Exception as exc:
                texts.append(f"[Attachment skipped: {exc}]")
                continue
            if up:
                uploads.append(up)
    return role, "\n\n".join(texts), uploads


def part_identity(part: object) -> str:
    """Cheap file identity (no downloads) for conversation fingerprinting."""
    if not isinstance(part, dict):
        return ""
    for key in ("filename", "file_id", "file_data", "url"):
        val = part.get(key)
        if isinstance(val, str) and val:
            return f"{key}={val[:160]}"
    for key in ("file", "image_url", "image"):
        nested = part.get(key)
        if isinstance(nested, dict):
            for sub in ("filename", "file_id", "url"):
                val = nested.get(sub)
                if isinstance(val, str) and val:
                    return f"{key}.{sub}={val[:160]}"
        elif isinstance(nested, str) and nested:
            return f"{key}={nested[:160]}"
    return part.get("type", "")


async def openai_messages_to_prompt(
    http: httpx.AsyncClient, messages: list[dict]
) -> tuple[str, list[tuple[UploadFile, str]]]:
    """Flatten one OpenAI messages array into a single prompt turn + file uploads.

    Multi-turn across requests is NOT rebuilt here: the Gemini ChatSession already
    holds history server-side. System/developer messages become a [System] preamble
    on the first turn of a conversation.
    """
    texts: list[str] = []
    uploads: list[tuple[UploadFile, str]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prefix = "" if role in ("user", "") else f"[{role.capitalize()}] "

        if isinstance(content, str):
            if content.strip():
                texts.append(f"{prefix}{content}" if prefix else content)
            continue

        for part in content or []:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype in ("text", "input_text", "output_text"):
                t = part.get("text", "")
                if t.strip():
                    texts.append(f"{prefix}{t}" if prefix else t)
            elif ptype == "refusal":
                texts.append(f"{prefix}[refusal] {part.get('refusal', '')}")
            else:
                up = await _part_to_file(http, part)
                if up:
                    uploads.append(up)

        for att in msg.get("attachments") or []:
            if isinstance(att, dict):
                up = await _part_to_file(http, {"type": "input_file", **att})
                if up:
                    uploads.append(up)

    prompt = "\n\n".join(texts) or " "
    return prompt, uploads


async def responses_input_to_prompt(
    http: httpx.AsyncClient, input: object
) -> tuple[str, list[tuple[UploadFile, str]]]:
    """Same flattening for Responses API input (string or item list)."""
    if isinstance(input, str):
        return input, []
    messages: list[dict] = []
    for item in input or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type", "message")
        if itype == "message":
            messages.append(
                {"role": item.get("role", "user"), "content": item.get("content", "")}
            )
        elif itype in ("image_generation_call",):
            messages.append({"role": "user", "content": "Generate an image."})
    return await openai_messages_to_prompt(http, messages)
