# Repository Guidelines

## Project Overview

OpenAI-compatible FastAPI front for `gemini.google.com` (no browser).
Translates Chat Completions, Responses, Files, Images, Models APIs to
`gemini-webapi` `ChatSession` calls over direct HTTPS (`curl_cffi`).
Base URL: `http://127.0.0.1:28407/v1`. Health: `GET /health` →
`{"status":"ok","accounts":2,"freeimage":true}`.

## Architecture & Data Flow

- **Shim, not a framework:** `server.py` (FastAPI routes) → `content.py`
  (parts → prompt + staged files) → `accounts.py` (`GeminiClient` pool) →
  native Gemini `ChatSession` → `freeimage.py` (generated-image re-host).
- **Persistence (SQLite, `gem2oai.db`):** `store.py:connect` (WAL,
  `busy_timeout`, shared schema) → `conversations.py:SessionStore`
  (sessions/aliases write-through, responses write-through + lazy rehydrate)
  and `content.py:UploadStore` (file bytes write-through, `bind` in
  lifespan). Nothing is lost on restart: chains and `/v1/files` ids survive.
- **Continuation without replay:** each conversation resumes server-side
  Gemini history via metadata `[cid, rid, rcid]` in `conversations.py`
  `SessionStore` (`key/alias → {account, metadata}`, response objects by id,
  cap-evicted, `asyncio.Lock`-guarded). Chained requests send only the new
  turn; history-prefix replay dedupes by content fingerprint.
- **Accounts:** N Netscape cookie jars in `accounts.txt` → round-robin
  `AccountPool.pick()` with 3-failure cooldown, sticky per conversation key,
  one `asyncio.Lock` per account serializing its turns.
- **Multimodal in:** `content.py:message_to_turn` → `(role, text, uploads)`;
  every non-text part goes through `_part_to_file` → `_as_upload`, which
  stages bytes to real temp paths (`/tmp/gem2oai-uploads/<id>_<name>.<ext>`).
  Required: `gemini-webapi` derives filename AND content-type from the path
  and ignores `BytesIO.name` (in-memory arrived as `input_*.txt`/`text/plain`).
  Callers must pass `str(up.path)` as `files=` and `up.cleanup()` in `finally`.
- **Image out:** `GeneratedImage.save()` → `IMG_TMP` bytes →
  `freeimage.upload_png` → prefer `display_url` (direct CDN) over `url`
  (viewer page) → `![Generated Image N](url)` markdown; `None` on missing
  key/failure so callers fall back to the Gemini URL.
- **Models:** `_resolve_model` maps `gemini-3.8-flash` → per-account registry
  flash; `-thinking` / `-extended-thinking` / `-et` suffix sets
  `extended_thinking=True`. `gemini-pro`, `gemini-flash-lite` mapped.
- **Tokens:** `len // 4` estimates; extended thoughts → `reasoning_content`.

## Key Directories

Flat root — no `src/`, `tests/`, or `docs/`:
- `.` — all 7 modules + `README.md`, `restart.sh`, `accounts.txt`, `.env`,
  `gem2oai.db`, `server.log`, `server.pid`.
- `.venv/` — uv-managed CPython 3.12, isolated (gitignored).
- `__pycache__/` — gitignored.
- `/tmp/gem2oai-uploads/` — staged upload temp files (auto-cleaned per turn).
- `/tmp/gem2oai-imgs/` (`IMG_TMP`) — generated-image download staging.

## Development Commands

No build step, no package manifest, no Makefile/CI.

```bash
uv venv .venv
uv pip install -p .venv/bin/python fastapi "uvicorn[standard]" httpx python-multipart gemini-webapi
printf 'FREEIMAGE_API_KEY=<key>\n' > .env   # never commit
.venv/bin/python server.py                  # foreground, port 28407
./restart.sh                                # kill + nohup restart, polls /health (PORT=28407 ./restart.sh to override)
tail -f server.log                          # follow logs (script prints this hint + base URL)
curl -s http://127.0.0.1:28407/health
curl -s http://127.0.0.1:28407/v1/models
```

`restart.sh` order: `pkill -f` both server patterns → `fuser -k 28407/tcp`
→ poll `ss -ltn` ≤10s → `nohup .venv/bin/python server.py >>server.log 2>&1 &`
→ write `server.pid` → poll `/health` ≤120s → print health + base URL + tail
hint; on death/timeout `tail -n 30 server.log`, exit 1.

## Code Conventions & Common Patterns

- **Naming:** snake_case modules/functions (`message_to_turn`, `_part_to_file`,
  `_run_turns`); `UpperCamel` classes (`AccountPool`, `SessionStore`,
  `UploadFile`, `UploadStore`); UPPER_SNAKE env (`PORT`, `FREEIMAGE_API_KEY`,
  `GEMINI_ACCOUNTS_FILE`); `_`-prefixed internals (`_pool`, `_sessions`,
  `_http`, `_err`, `_sse`, `_fingerprint`, `_norm_input_item`).
- **Async:** `async def` handlers throughout; per-account `asyncio.Lock`
  held for whole conversation update (`_run_turns` uses
  `async with pool.lock_for(account)`; `_run_turns_stream` manual
  acquire/release). `asyncio.gather` for parallel image re-hosts.
- **Uploads:** always `files = [str(up.path) for up, _ in uploads] or None`,
  then `for up, _ in uploads: up.cleanup()` in `finally` — never pass
  `BytesIO`, never skip cleanup.
- **Errors:** `_err(status, message, code)` →
  `{"error": {"message", "type", "code"}}`; `_map_exception` maps
  UsageLimit→429, timeout→504, auth→502, else 500. Failed URL downloads
  degrade to `[Attachment skipped: ...]` text inline, never 500.
- **Streaming:** chat SSE via `_sse()` + `data: [DONE]`; responses SSE via
  `response.created` / `response.output_text.delta` / `response.completed`
  (`response.failed` on error).
- **State access:** handlers start `pool, sessions, http =
  _pool(req), _sessions(req), _http(req)` (lifespan-injected `app.state`).
- **Shell (`restart.sh`):** `#!/usr/bin/env bash` + `set -u` (no `set -e`;
  explicit `|| true`, explicit `exit 1`); `DIR` from `BASH_SOURCE[0]`;
  `==>` progress / `!!` failure echo prefixes.
- **Secrets:** `config._load_dotenv()` fills env from `.env` without
  overwriting real env; `freeimage_api_key()` accessor; never hardcode keys.

## Important Files

- `server.py` — entry point; all `/v1/*` routes, `_run_turns[_stream]`,
  chaining (`_chat_turns`, `_responses_turns`), `_model_list`; run via
  `uvicorn.run(app, host="0.0.0.0", port=PORT)` in `__main__`.
- `content.py` — `message_to_turn`, `openai_messages_to_prompt`,
  `responses_input_to_prompt`, `_part_to_file`, `_as_upload`, `UploadFile`
  (staged path + `cleanup()`), `UploadStore` (Files API, SQLite-backed),
  `part_identity`.
- `accounts.py` — `load_account_cookies`, `AccountPool` (pick/lock_for/report,
  `init_all`/`close_all`, `models`, `resolve`, `display_slugs`).
- `conversations.py` — `SessionStore` (SQLite-backed: `get`/`get_or_new`/
  `persist`/`link`, `save_response`/`get_response`),
  `SessionState{account, metadata, lock}`.
- `store.py` — `connect(path)` (SQLite WAL schema: sessions/aliases/
  responses/files); shared by both stores.
- `config.py` — `BASE_DIR`, `ACCOUNTS_FILE` (`GEMINI_ACCOUNTS_FILE`),
  `PORT` (default `28407`), `DB_PATH` (`GEMINI_DB_PATH`, default
  `gem2oai.db`), `_load_dotenv`, `freeimage_api_key`.
- `freeimage.py` — `upload_png(http, data, filename)`, `ENDPOINT =
  https://freeimage.host/api/1/upload`.
- `README.md` — sole doc; install/run/endpoints/secrets reference.
- `restart.sh` — only helper script; entire ops surface.
- `.env` — `FREEIMAGE_API_KEY`, `PORT` (gitignored, never commit/read aloud).
- `accounts.txt` — cookie jars, secret-equivalent and gitignored (see
  `accounts.txt.example` for the committed template): do not commit or paste contents.
- `server.log` / `server.pid` — nohup output + live PID (gitignored runtime state).
- `.gitignore` — `__pycache__/`, `.venv/`, `.env`, `*.pyc`, `temp/`,
  `gem2oai-imgs/`, `gem2oai.db*`, `accounts.txt`, `server.log`, `server.pid`.

## Runtime/Tooling Preferences

- **Runtime:** CPython 3.12; interpreter is ALWAYS `$DIR/.venv/bin/python`,
  never system python (see `.venv/pyvenv.cfg`, isolated).
- **Package manager:** `uv` only (`uv venv`, `uv pip install -p ...`);
  no `pyproject.toml` / `setup.py` / `requirements*.txt` / lockfile — do not
  invent one.
- **Server:** `uvicorn`, host `0.0.0.0`, default port `28407` (`PORT` env
  overrides). Key deps: `fastapi`, `uvicorn[standard]`, `httpx`,
  `python-multipart`, `gemini-webapi` (`curl_cffi`, `orjson`, `loguru`,
  `pydantic` transitive).
- **Constraints:** no browser tooling anywhere in the path; no test configs
  to honor; keep changes to existing flat modules, no new dirs.
- **Lint/typecheck:** Always use https://docs.astral.sh/ruff/ with everything enabled and https://docs.astral.sh/ty/ with everything enabled, then fix all of the issues. Make sure to actually fix all of the issues instead of suppressing them.

## Testing & QA

- **Frameworks:** none. No `tests/`, `test_*.py`, `conftest.py`, pytest/unittest
  config, lint/format (ruff/black/mypy), or CI — verified by exhaustive glob.
  Do not claim pytest/coverage exists.
- **Verify manually:**
  ```bash
  curl -s http://127.0.0.1:28407/health   # {"status":"ok",...}
  curl -s http://127.0.0.1:28407/v1/models
  curl -s http://127.0.0.1:28407/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemini-3.8-flash","messages":[{"role":"user","content":"Reply with exactly: STILL-OK"}]}'
  ```
- **Image-path regression:** send a base64 `image_url` part and confirm the
  reply describes image content (not `IHDR`/`IDAT` text); check
  `/tmp/gem2oai-uploads/` is empty afterwards (cleanup ran).
- **Coverage expectations:** none stated or enforced.
