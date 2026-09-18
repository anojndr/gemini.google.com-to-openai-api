# Copyright 2026 gem2oai contributors.
"""Settings: env/.env loading. NEVER hardcode secrets here."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    """Load KEY=value pairs from .env without overwriting real environment."""
    env_file = BASE_DIR / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        name, val = key.strip(), value.strip().strip("'").strip('"')
        if name and name not in os.environ:
            os.environ[name] = val


_load_dotenv()

ACCOUNTS_FILE = Path(os.environ.get("GEMINI_ACCOUNTS_FILE", BASE_DIR / "accounts.txt"))
# Namespaced env first; generic PORT last so a sibling bridge's export can't rebind us.
PORT = int(os.environ.get("GEMINI_PORT", os.environ.get("PORT", "28407")))
DB_PATH = Path(os.environ.get("GEMINI_DB_PATH") or BASE_DIR / "gem2oai.db")


def freeimage_api_key() -> str:
    """Return the freeimage.host API key from the environment."""
    return os.environ.get("FREEIMAGE_API_KEY", "").strip()
