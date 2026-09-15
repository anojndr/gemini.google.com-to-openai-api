"""freeimage.host upload for Gemini-generated images. Key from env only."""

from __future__ import annotations

import httpx

from config import freeimage_api_key

ENDPOINT = "https://freeimage.host/api/1/upload"


async def upload_png(http: httpx.AsyncClient, data: bytes, filename: str) -> str | None:
    """Upload raw PNG/JPEG bytes; return a direct image URL, None on failure.

    Prefers `display_url` (direct CDN file) over `url` (viewer page) so the
    link works as an <img> src outside freeimage.host.
    """
    key = freeimage_api_key()
    if not key:
        return None
    try:
        r = await http.post(
            ENDPOINT,
            data={"key": key, "action": "upload", "format": "json"},
            files={"source": (filename, data, "image/png")},
        )
        r.raise_for_status()
        body = r.json()
        if isinstance(body, dict):
            img = body.get("image") or {}
            if isinstance(img, dict):
                return img.get("display_url") or img.get("url") or body.get("url")
            return body.get("url")
        return None
    except Exception:
        return None
