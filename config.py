"""Settings: env/.env loading. NEVER hardcode secrets here."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ACCOUNTS_FILE = Path(os.environ.get("GEMINI_ACCOUNTS_FILE", BASE_DIR / "accounts.txt"))
PORT = int(os.environ.get("PORT", "28407"))


def _load_dotenv() -> None:
    env_file = BASE_DIR / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def freeimage_api_key() -> str:
    return os.environ.get("FREEIMAGE_API_KEY", "").strip()
