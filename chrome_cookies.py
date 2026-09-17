# Copyright 2026 gem2oai contributors.
"""Live cookie source: Google auth cookies straight from local Chrome profiles.

Chrome is the default browser on this PC and stays logged in to both Google
accounts (profile `fogent` = account 1, profile `olleat` = account 2). Its
cookie store refreshes itself whenever the browser is used, so re-exporting
from disk at startup and on every sync interval means accounts.txt (and
degraded live clients) never go stale: no manual re-paste, cookies never
expire.

Layout: file blocks map positionally to profiles (block 1 = first profile).
Profile names resolve through Chrome's `Local State` info_cache, so a
directory rename keeps working; a name matching no profile is treated as a
directory name directly.

Decryption: Linux Chrome encrypts cookie values with AES-128-CBC, key =
PBKDF2-SHA1 of the `Chrome Safe Storage` keyring secret (salt `saltysalt`,
1 iteration), IV = 16 spaces. The plaintext carries a 32-byte artifact
prefix followed by PKCS#7-padded value.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

_BLOCK_RE = re.compile(r"```\n(.*?)```", re.DOTALL)

_CHROME_EPOCH_DELTA_US = 11644473600000000
_KEY_SALT = b"saltysalt"
_KEY_ITERATIONS = 1
_CBC_IV = b" " * 16
_PLAINTEXT_PREFIX = 32
_AES_BLOCK_SIZE = 16
_JAR_COLUMNS = 7
_SECRET_TOOL = "/usr/bin/secret-tool"

_JAR_HOSTS = (".google.com", ".gemini.google.com")
_POOL_HOST = ".google.com"

CookieRow = dict[str, object]


def _chrome_dir() -> Path:
    """Return the Chrome user-data dir, overridable via GEMINI_CHROME_DIR."""
    override = os.environ.get("GEMINI_CHROME_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "google-chrome"


def _profile_names() -> list[str]:
    """Return Chrome profile display names, one per accounts.txt block."""
    raw = os.environ.get("GEMINI_CHROME_PROFILES", "fogent,olleat")
    return [p.strip() for p in raw.split(",") if p.strip()]


def profile_dirs() -> list[str]:
    """Resolve profile display names to Chrome directory names via Local State."""
    base = _chrome_dir()
    try:
        state = json.loads((base / "Local State").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    cache = state.get("profile", {}).get("info_cache", {})
    by_name = {
        info.get("name"): dirname
        for dirname, info in cache.items()
        if isinstance(info, dict) and info.get("name")
    }
    return [by_name.get(name, name) for name in _profile_names()]


def _storage_key() -> bytes:
    """Derive the Linux Chrome AES key from the keyring Safe Storage secret."""
    try:
        proc = subprocess.run(
            [
                _SECRET_TOOL,
                "lookup",
                "server",
                "Chrome Keys",
                "user",
                "Chrome Safe Storage",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        msg = f"secret-tool unavailable: {exc}"
        raise RuntimeError(msg) from exc
    secret = proc.stdout.strip()
    if not secret:
        msg = "Chrome Safe Storage secret not found in keyring"
        raise RuntimeError(msg)
    return hashlib.pbkdf2_hmac(
        "sha1",
        secret.encode(),
        _KEY_SALT,
        _KEY_ITERATIONS,
        _AES_BLOCK_SIZE,
    )


def _decrypt_value(value: str, blob: bytes, key: bytes) -> str:
    """Return the plaintext cookie value from a Chrome cookies row."""
    if not blob:
        return value
    if blob[:3] not in (b"v10", b"v11"):
        msg = f"unsupported cookie encryption prefix: {blob[:3]!r}"
        raise ValueError(msg)
    decryptor = Cipher(algorithms.AES(key), modes.CBC(_CBC_IV)).decryptor()
    padded = decryptor.update(blob[3:]) + decryptor.finalize()
    plain = padded[_PLAINTEXT_PREFIX:]
    pad = plain[-1] if plain else 0
    if 1 <= pad <= _AES_BLOCK_SIZE and plain.endswith(bytes([pad]) * pad):
        plain = plain[:-pad]
    return plain.decode("utf-8")


def _snapshot_db(src: Path) -> Path:
    """Online-backup a (possibly Chrome-locked) SQLite DB to a temp copy."""
    tmp = Path(tempfile.mkstemp(prefix="gem2oai-chrome-", suffix=".db")[1])
    src_con = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=10)
    try:
        dst_con = sqlite3.connect(str(tmp))
        try:
            src_con.backup(dst_con)
        finally:
            dst_con.close()
    finally:
        src_con.close()
    return tmp


def read_chrome_jar(profile_dir: str) -> list[CookieRow]:
    """Read and decrypt Google cookies for one Chrome profile directory."""
    src = _chrome_dir() / profile_dir / "Cookies"
    if not src.is_file():
        msg = f"Chrome cookie store not found: {src}"
        raise FileNotFoundError(msg)
    key = _storage_key()
    tmp = _snapshot_db(src)
    try:
        con = sqlite3.connect(str(tmp))
        try:
            rows = con.execute(
                "SELECT host_key, path, is_secure, is_httponly,"
                " expires_utc, name, value, encrypted_value"
                " FROM cookies WHERE host_key IN (?, ?)",
                _JAR_HOSTS,
            ).fetchall()
        finally:
            con.close()
    finally:
        tmp.unlink(missing_ok=True)
    jar: list[CookieRow] = []
    for host, path, secure, httponly, expires_us, name, value, blob in rows:
        try:
            plain = _decrypt_value(value or "", blob or b"", key)
        except (ValueError, UnicodeDecodeError):
            continue
        if not plain:
            continue
        expires = (
            str(max(0, (int(expires_us) - _CHROME_EPOCH_DELTA_US) // 1_000_000))
            if expires_us
            else "0"
        )
        jar.append(
            {
                "domain": str(host),
                "path": str(path or "/"),
                "secure": bool(secure),
                "httponly": bool(httponly),
                "expires": expires,
                "name": str(name),
                "value": plain,
            },
        )
    jar.sort(key=lambda c: (str(c["domain"]), str(c["name"])))
    return jar


def jar_dict(profile_dir: str) -> dict[str, str]:
    """Return .google.com name -> value pairs for one Chrome profile."""
    return {
        str(c["name"]): str(c["value"])
        for c in read_chrome_jar(profile_dir)
        if c["domain"] == _POOL_HOST
    }


def _block_cookie_key(line: str) -> tuple[str, str] | None:
    """Return the (domain, name) key for a Netscape jar line, else None."""
    rest = line.strip()
    if rest.startswith("#HttpOnly"):
        rest = rest[len("#HttpOnly") :].lstrip()
    elif rest.startswith("#"):
        return None
    parts = rest.split()
    if len(parts) < _JAR_COLUMNS or not parts[5]:
        return None
    return parts[0], parts[5]


def _format_line(cookie: CookieRow) -> str:
    """Format one Netscape jar line from a Chrome cookie row."""
    domain = str(cookie["domain"])
    dotted = "TRUE" if domain.startswith(".") else "FALSE"
    secure = "TRUE" if cookie["secure"] else "FALSE"
    prefix = "#HttpOnly " if cookie["httponly"] else ""
    return (
        f"{prefix}{domain}\t{dotted}\t{cookie['path']}\t{secure}\t"
        f"{cookie['expires']}\t{cookie['name']}\t{cookie['value']}"
    )


def refresh_accounts_from_chrome(path: str | Path) -> int:
    """Refresh accounts.txt jar blocks from live Chrome profiles.

    Blocks map positionally to Chrome profiles (block 1 = first profile).
    Values refresh from Chrome; Chrome names missing from a block are
    appended; file-only lines (stale exports, comments) pass through
    untouched, so nothing that works today is ever deleted. Atomic write,
    no-op when every block already matches. Returns updated blocks.
    """
    file = Path(path)
    try:
        raw = file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    dirs = profile_dirs()
    new_raw = raw
    updated = 0
    for match, profile_dir in zip(
        _BLOCK_RE.finditer(raw),
        dirs,
        strict=False,
    ):
        try:
            jar = read_chrome_jar(profile_dir)
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            continue
        by_key = {(str(c["domain"]), str(c["name"])): c for c in jar}
        if "__Secure-1PSID" not in {name for _, name in by_key}:
            continue
        seen: set[tuple[str, str]] = set()
        out = []
        for ln in match.group(1).split("\n"):
            key = _block_cookie_key(ln)
            row = by_key.get(key) if key is not None else None
            if key is None or row is None:
                out.append(ln)
                continue
            seen.add(key)
            out.append(_format_line(row))
        missing = sorted(
            (k for k in by_key if k not in seen),
            key=lambda k: (k[0], k[1]),
        )
        stripped = "\n".join(out).rstrip("\n")
        extra = "".join("\n" + _format_line(by_key[k]) for k in missing)
        new_block = stripped + extra + ("\n" if match.group(1).endswith("\n") else "")
        if new_block == match.group(1):
            continue
        new_raw = new_raw.replace(match.group(0), "```\n" + new_block + "```", 1)
        updated += 1
    if not updated:
        return 0
    try:
        mode = file.stat().st_mode & 0o777
        tmp = file.with_name(file.name + ".tmp")
        tmp.write_text(new_raw, encoding="utf-8")
        tmp.chmod(mode)
        tmp.replace(file)
    except OSError:
        return 0
    return updated
