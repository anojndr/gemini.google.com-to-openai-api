# gemini.google.com → OpenAI-compatible API

Direct-HTTPS bridge (no browser): `gemini-webapi` (curl_cffi) → FastAPI on port **28407**.

## Run

```bash
uv venv .venv
uv pip install -p .venv/bin/python fastapi "uvicorn[standard]" httpx \
  python-multipart gemini-webapi cryptography redis
cp .env.example .env            # then put your freeimage.host key in .env (never commit)
cp accounts.txt.example accounts.txt   # then paste your Gemini cookie jars (never commit)
.venv/bin/python server.py      # or: ./restart.sh (background + /health check)
```

Base URL for OpenAI clients: `http://127.0.0.1:28407/v1`

`GET /health` → `{"status":"ok","accounts":2,"auth":"ok|degraded","freeimage":true,"redis":{...}}`
(`auth` reflects live Gemini session state; `degraded` means cookies expired
and only guest generation is available; `redis.status` is `ok|down|disabled`
with latency, hit-ratio, ops/sec, and memory when configured).

## Endpoints (OpenAI-compatible)

- `POST /v1/chat/completions` (+ `/chat/completions`) — `stream:true` supported
- `POST /v1/responses` (+ `/responses`), `GET /v1/responses/{id}`
  - `previous_response_id` chaining, `conversation` param, streaming SSE
- `GET /v1/models` — registry models + `gemini-3.8-flash*` aliases
- `POST /v1/files`, `GET /v1/files`, `GET /v1/files/{id}`,
  `DELETE /v1/files/{id}`, `GET /v1/files/{id}/content`
- `POST /v1/images/generations` — Gemini image → freeimage.host URL
- `GET /health`, `GET /v1/conversations/{id}`

## Notes

- **Accounts**: every ```` ``` ```` cookie-jar block in `accounts.txt` is an
  account (unbounded; currently 2). Round-robin + 3-strike cooldown failover,
  sticky per conversation. Block 1 ↔ Chrome profile `fogent`, block 2 ↔
  profile `olleat` (`GEMINI_CHROME_PROFILES` overrides; `GEMINI_CHROME_DIR`
  overrides the user-data dir).
- **Multi-turn**: native Gemini `ChatSession` per conversation; chained
  requests send only the new turn. History-prefix replay also dedupes via
  fingerprinting.
- **Persistence**: conversations, aliases, response objects, and uploaded
  files live in SQLite (`gem2oai.db`, WAL mode; `GEMINI_DB_PATH` overrides)
  and survive restarts — chains (`conversation_id`, `previous_response_id`,
  fingerprint prefixes) and `/v1/files` ids keep working after `./restart.sh`.
  Set `REDIS_URL` (or `GEMINI_REDIS_URL`, which wins) to enable the shared
  tier, e.g. `redis://127.0.0.1:6379/0`; unset or empty (or unreachable)
  stays SQLite-only. Every write then also dual-writes to Redis
  (`gem2oai:session/response/file/alias:*` + recency ZSets) so a second
  process or a restart sees the same rows; Redis failures degrade to SQLite
  per call. `/health` reports `redis` liveness, latency, hit-ratio, and memory.
- **Auth degradation**: when cookies expire, requests for unavailable models
  fall back to the guest-selectable model (or Google's default) instead of
  502; guest-era continuations that fail with a resume timeout retry once
  fresh. File upload and image generation require an authenticated session
  and still 502 until cookies are refreshed.
- **Cookie self-sync**: cookies never expire by themselves — at startup and
  every 10 min the server re-exports Google cookies from the live Chrome
  profiles straight into `accounts.txt` (positional block ↔ profile match),
  then writes live (rotated) client values back over the file so
  server-side 1PSIDTS rotation always wins over older Chrome values. A
  degraded (non-AVAILABLE) account is rebuilt in place from fresh Chrome
  cookies on the same interval, under its per-account lock, so it heals
  without a restart. Per-block matching stays by `__Secure-1PSID` for the
  live-value pass (blocks sharing one PSID refresh identically), with
  missing names appended. Cookie cache is project-local
  (`.gemini-cookie-cache/`, gitignored, cleared at startup) so parallel
  checkouts and probe scripts can never poison it with stale sessions.
  Manual re-paste is only for a logged-out browser.
- **Files**: all part types (`image_url`, `input_image`, `input_file`,
  `file`, data-URLs, attachments, `/v1/files` ids) for images/JSON/txt/py/…;
  bytes are staged to real temp paths (`/tmp/gem2oai-uploads/`, cleaned up
  after each turn) because gemini-webapi derives filename AND content-type
  from the path — `BytesIO.name` is ignored, so in-memory buffers arrived
  as `input_*.txt`/`text/plain` (Gemini saw raw IHDR/IDAT chunks).
  Failed URL downloads degrade to `[Attachment skipped]` text, never 500.
- **Image output**: `GeneratedImage.save()` bytes → freeimage.host
  (`display_url` direct CDN link) → `![Generated Image N](url)` markdown.
  Verified live: "generate an image of a cat" → tabby-cat JPEG (500×273).
- **Secrets**: `FREEIMAGE_API_KEY` from env/`.env` only; never hardcoded.
